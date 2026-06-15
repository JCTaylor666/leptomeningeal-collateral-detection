#!/usr/bin/env python3
"""LMC inference visualization web demo.

Drag a DSA frame into the browser, press Start, and watch every pipeline
stage render as soon as the backend finishes computing it:

    DSA ─▶ vessel seg ─▶ graph build ─▶ graph branch (GNN) ─┐
                      └▶ masked input ─▶ pixel branch ───────┴▶ fusion ─▶ result

The backend reuses the exact same runners as `scripts/infer_one.py`
(`EndToEndInference`), calling each stage in order and streaming its
visualization to the page over a single HTTP response (SSE framing).

Run:
    export LMC_DINOV3_REPO=/path/to/dinov3
    python demo/app.py                       # serves http://127.0.0.1:5000
    # options: --host --port --ckpt_root --dino_repo_dir --device

Heavy model weights load lazily on the first inference (5-fold nnU-Net ×2
+ DINOv3×GAT ×5), so the first run is slower; later runs reuse them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional

# nnU-Net reads these env vars at import time; point them at a scratch dir so
# the (training-only) raw/preprocessed/results paths don't warn. Inference
# loads checkpoints directly from ckpt/. Must run BEFORE importing nnunetv2
# (transitively via the runners). Pre-existing shell values are preserved.
_NNUNET_PLACEHOLDER = tempfile.gettempdir()
os.environ.setdefault("nnUNet_raw", _NNUNET_PLACEHOLDER)
os.environ.setdefault("nnUNet_preprocessed", _NNUNET_PLACEHOLDER)
os.environ.setdefault("nnUNet_results", _NNUNET_PLACEHOLDER)

import base64
import io

import numpy as np
from PIL import Image

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
for p in (str(DEMO_DIR), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402
from flask import stream_with_context  # noqa: E402

import viz  # local: demo/viz.py  # noqa: E402

from lmc.config import InferenceConfig  # noqa: E402
from lmc.fusion import fuse  # noqa: E402
from lmc.graph_build import build_graph_from_mask, sort_nodes_by_id  # noqa: E402
from lmc.pipeline import EndToEndInference  # noqa: E402
from lmc.pixel_branch.runner import pool_pixel_prob_to_nodes  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration (mirrors scripts/infer_one.py resolution rules).
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve(repo_root: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p).resolve()


def build_config(args: argparse.Namespace) -> InferenceConfig:
    cfg_dict = _load_yaml(args.config)
    ckpt_root = _resolve(REPO_ROOT, args.ckpt_root or cfg_dict.get("ckpt_root", "ckpt"))

    dino_raw = args.dino_repo_dir or os.environ.get("LMC_DINOV3_REPO") or cfg_dict.get("dino_repo_dir")
    if dino_raw is None or str(dino_raw) in {"", "/path/to/dinov3"}:
        raise SystemExit(
            "ERROR: DINOv3 source-repo path is not set.\n"
            "  Set one of: --dino_repo_dir /path/to/dinov3 | export LMC_DINOV3_REPO=... | configs/inference_default.yaml"
        )
    dino_repo = Path(dino_raw)
    if not dino_repo.is_dir():
        raise SystemExit(f"ERROR: DINOv3 repo not found: {dino_repo}")

    return InferenceConfig(
        ckpt_root=ckpt_root,
        dino_repo_dir=dino_repo,
        nnunet_lambda=float(cfg_dict.get("nnunet_lambda", 0.77)),
        folds=tuple(cfg_dict.get("folds", [0, 1, 2, 3, 4])),
        device=str(args.device or cfg_dict.get("device", "cuda")),
        amp=bool(cfg_dict.get("amp", True)),
    )


# --------------------------------------------------------------------------- #
# Global pipeline (lazy weight load) + single-run lock.
# --------------------------------------------------------------------------- #
CFG: Optional[InferenceConfig] = None
PIPE: Optional[EndToEndInference] = None
INFER_LOCK = threading.Lock()

app = Flask(__name__, static_folder=None)


def _sse(stage: str, **payload) -> str:
    return "data: " + json.dumps({"stage": stage, **payload}) + "\n\n"


def _decode_image(data_uri: str) -> np.ndarray:
    """data:image/...;base64,XXXX  →  (H, W) uint8 grayscale."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _resize_prob(prob: np.ndarray, target_hw) -> np.ndarray:
    """Match pipeline.run_single_frame's pixel-prob resize (bilinear, uint16)."""
    if prob.shape == tuple(target_hw):
        return prob
    pp = Image.fromarray((prob * 65535).astype(np.uint16))
    pp = pp.resize((int(target_hw[1]), int(target_hw[0])), resample=Image.BILINEAR)
    return np.asarray(pp, dtype=np.float32) / 65535.0


def run_stream(image_uint8: np.ndarray, caseid: str, lam: float, threshold: float) -> Iterable[str]:
    """Generator: run the pipeline stage-by-stage, yielding SSE events.

    Mirrors EndToEndInference.run_single_frame (ensemble path) but emits a
    visualization after each stage instead of returning a single dict.
    """
    if not INFER_LOCK.acquire(blocking=False):
        yield _sse("error", message="Another inference is already running. Please wait.")
        return
    try:
        folds = tuple(CFG.folds)
        yield _sse(
            "meta",
            caseid=caseid,
            num_folds=len(folds),
            lam=float(lam),
            threshold=float(threshold),
            colorbar=viz.colorbar(),
        )
        yield _sse("result", step="input", image=viz.encode_image(image_uint8),
                   caption=f"{caseid} · {image_uint8.shape[1]}×{image_uint8.shape[0]}")

        image_float = image_uint8.astype(np.float32) / 255.0

        # 1) Vessel segmentation -------------------------------------------- #
        yield _sse("status", step="vessel_seg", message=f"nnU-Net + clDice · {len(folds)}-fold ensemble…")
        mask_uint8, _ = PIPE.vessel_seg.predict_mask(image_uint8, caseid=caseid, return_probability=False)
        vessel_mask_float = (mask_uint8 > 0).astype(np.float32)
        vessel_px = int((mask_uint8 > 0).sum())
        yield _sse("result", step="vessel_seg", image=viz.vessel_overlay(image_uint8, mask_uint8),
                   caption=f"{vessel_px:,} vessel px")

        # 2) Graph construction --------------------------------------------- #
        yield _sse("status", step="graph", message="line-graph: nodes = vessel segments…")
        graph = build_graph_from_mask(mask_uint8, source_name=f"{caseid}.png")
        if graph is None:
            yield _sse("error", message="Empty vessel mask — no graph could be built.")
            return
        nodes = sort_nodes_by_id(graph.get("nodes", []))
        graph_for_branch = {**graph, "nodes": nodes}
        yield _sse("result", step="graph", image=viz.graph_overlay(image_uint8, graph),
                   caption=f"{len(nodes)} nodes · {graph.get('num_edges', 0)} edges")

        # 3) Masked input (what the pixel branch sees) ---------------------- #
        yield _sse("result", step="masked", image=viz.masked_input(image_uint8, mask_uint8),
                   caption="image × (vessel mask > 0)")

        # 4) Pixel branch --------------------------------------------------- #
        yield _sse("status", step="pixel_branch", message=f"vessel-masked nnU-Net · {len(folds)}-fold…")
        pixel_prob = PIPE.pixel_branch.predict_prob(
            image_uint8, caseid=caseid, folds=folds, vessel_mask=mask_uint8
        )
        target_hw = tuple(graph_for_branch.get("edge_id_img_shape", image_float.shape))
        pixel_prob = _resize_prob(pixel_prob, target_hw)
        p_nn = pool_pixel_prob_to_nodes(pixel_prob, nodes)
        yield _sse("result", step="pixel_branch", image=viz.pixel_heatmap(image_uint8, pixel_prob),
                   caption=f"p_nn  mean={float(p_nn.mean()):.3f}  max={float(p_nn.max()):.3f}")

        # 5) Graph branch --------------------------------------------------- #
        yield _sse("status", step="graph_branch", message=f"DINOv3 ViT-L/16 + GAT · {len(folds)}-fold…")
        p_gnn = PIPE.graph_branch.forward_one_frame_ensemble(
            graph_for_branch, image=image_float, vessel_mask=vessel_mask_float, folds=folds
        )
        yield _sse("result", step="graph_branch", image=viz.node_score_overlay(image_uint8, nodes, p_gnn),
                   caption=f"p_gnn  mean={float(p_gnn.mean()):.3f}  max={float(p_gnn.max()):.3f}")

        # 6) Fusion --------------------------------------------------------- #
        yield _sse("status", step="fusion", message=f"p_fuse = λ·p_gnn + (1−λ)·p_nn  (λ={lam:g})")
        p_fuse = fuse(p_gnn, p_nn, lam=float(lam))
        yield _sse("result", step="fusion", image=viz.node_score_overlay(image_uint8, nodes, p_fuse),
                   caption=f"p_fuse  mean={float(p_fuse.mean()):.3f}  max={float(p_fuse.max()):.3f}")

        # 7) Decision ------------------------------------------------------- #
        positive = int((p_fuse >= threshold).sum())
        yield _sse("result", step="result", image=viz.final_overlay(image_uint8, nodes, p_fuse, threshold),
                   caption=f"{positive}/{len(nodes)} nodes ≥ τ={threshold:g}")

        yield _sse("done", num_nodes=len(nodes), positive=positive, lam=float(lam), threshold=float(threshold))
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        yield _sse("error", message=f"{type(exc).__name__}: {exc}")
    finally:
        INFER_LOCK.release()


# --------------------------------------------------------------------------- #
# Routes.
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(DEMO_DIR / "static", "index.html")


@app.route("/static/<path:fname>")
def static_files(fname: str):
    return send_from_directory(DEMO_DIR / "static", fname)


@app.route("/health")
def health():
    return jsonify(
        ckpt_root=str(CFG.ckpt_root),
        dino_repo=str(CFG.dino_repo_dir),
        device=CFG.device,
        folds=list(CFG.folds),
        lam=CFG.nnunet_lambda,
    )


@app.route("/infer", methods=["POST"])
def infer():
    body = request.get_json(force=True, silent=True) or {}
    image_b64 = body.get("image")
    if not image_b64:
        return jsonify(error="no image provided"), 400
    try:
        image_uint8 = _decode_image(image_b64)
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"could not decode image: {exc}"), 400

    caseid = str(body.get("caseid") or "upload").strip() or "upload"
    caseid = "".join(c for c in caseid if c.isalnum() or c in "._-") or "upload"
    lam = float(body.get("lam", CFG.nnunet_lambda))
    threshold = float(body.get("threshold", 0.5))

    gen = run_stream(image_uint8, caseid, lam, threshold)
    resp = Response(stream_with_context(gen), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"  # disable proxy buffering
    return resp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LMC inference visualization web demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "inference_default.yaml")
    ap.add_argument("--ckpt_root", type=Path, default=None)
    ap.add_argument("--dino_repo_dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    global CFG, PIPE
    args = parse_args()
    CFG = build_config(args)
    PIPE = EndToEndInference(CFG)  # cheap; weights load lazily on first stage
    print("=" * 64)
    print("LMC inference demo")
    print(f"  ckpt_root : {CFG.ckpt_root}")
    print(f"  dino_repo : {CFG.dino_repo_dir}")
    print(f"  device    : {CFG.device}   folds: {list(CFG.folds)}   λ: {CFG.nnunet_lambda}")
    print(f"  open      : http://{args.host}:{args.port}")
    print("=" * 64)
    # threaded=True so the streaming response and static assets coexist;
    # actual inference is serialized by INFER_LOCK.
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
