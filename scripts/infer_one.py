#!/usr/bin/env python3
"""End-to-end LMC collateral detection on a single DSA frame.

Feed one 2D DSA frame; the pipeline predicts a vessel mask, builds the
vessel-segment graph, runs the graph and pixel branches, fuses them, and
writes per-vessel-segment collateral probabilities.

Usage:
  python scripts/infer_one.py \
      --image PATH/sXX_fYY.png \
      --output_dir PATH/out \
      [--config configs/inference_default.yaml] \
      [--ckpt_root PATH/ckpt] \
      [--dino_repo_dir /path/to/dinov3] \
      [--mode ensemble | fold:N]               # default: ensemble
      [--lambda 0.77] \
      [--device cuda] \
      [--threshold 0.5]

Outputs (under --output_dir):
  <caseid>_pred.json          per-node {id, bbox, p_gnn, p_nn, p_fuse, hard_pred}
  <caseid>_overlay.png        DSA + graph + collateral highlights
  <caseid>_vessel_mask.png    binary vessel mask predicted by nnU-Net + clDice
  <caseid>_graph_pred.json    full predicted graph topology
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Suppress benign nnU-Net startup warnings ("nnUNet_raw is not defined ...").
# These env vars are only used by training utilities; single-frame inference
# loads checkpoints directly from `ckpt/` and does not touch nnUNet_raw /
# nnUNet_preprocessed / nnUNet_results paths. Set placeholders BEFORE the
# nnunetv2 import (which happens transitively via vessel_seg_runner /
# pixel_branch_runner). Pre-existing values in the user's shell are
# preserved via setdefault.
_NNUNET_PLACEHOLDER = tempfile.gettempdir()
os.environ.setdefault("nnUNet_raw", _NNUNET_PLACEHOLDER)
os.environ.setdefault("nnUNet_preprocessed", _NNUNET_PLACEHOLDER)
os.environ.setdefault("nnUNet_results", _NNUNET_PLACEHOLDER)

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lmc.pipeline import EndToEndInference  # noqa: E402
from lmc.config import InferenceConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LMC end-to-end inference (single frame)")
    ap.add_argument("--image", type=Path, required=True, help="DSA frame PNG")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "inference_default.yaml")
    ap.add_argument("--ckpt_root", type=Path, default=None)
    ap.add_argument("--dino_repo_dir", type=Path, default=None)
    ap.add_argument("--mode", type=str, default="ensemble", help="'ensemble' (default) or 'fold:N'")
    ap.add_argument("--lambda", dest="nnunet_lambda", type=float, default=None, help="Override fusion weight λ")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=0.5, help="Hard-decision threshold τ")
    return ap.parse_args()


def _load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing inference config: {path}")
    with open(path) as fh:
        import yaml

        return yaml.safe_load(fh)


def _resolve_path(repo_root: Path, value) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _draw_overlay(
    image_path: Path,
    graph: Dict,
    p_score: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")
    nodes = sorted(graph.get("nodes", []), key=lambda node: int(node["id"]))
    for node, score in zip(nodes, p_score):
        bbox = node["bbox"]
        y0, y1, x0, x1 = (int(v) for v in bbox)
        if score >= threshold:
            colour = (255, 70, 70, 220)
            for px in node.get("pixels", []):
                py, px_ = int(px[0]), int(px[1])
                draw.point((px_, py), fill=colour)
            draw.rectangle([x0, y0, x1, y1], outline=(255, 70, 70, 255), width=1)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=(70, 200, 220, 120), width=1)
    base.save(out_path)


def _per_node_records(
    graph: Dict,
    p_gnn: np.ndarray,
    p_nn: np.ndarray,
    p_fuse: np.ndarray,
    threshold: float,
) -> List[Dict]:
    nodes = sorted(graph.get("nodes", []), key=lambda node: int(node["id"]))
    out: List[Dict] = []
    for idx, node in enumerate(nodes):
        out.append(
            {
                "node_id": int(node["id"]),
                "bbox": [int(v) for v in node["bbox"]],
                "num_pixels": len(node.get("pixels", [])),
                "p_gnn": float(p_gnn[idx]),
                "p_nn": float(p_nn[idx]),
                "p_fuse": float(p_fuse[idx]),
                "hard_pred": int(p_fuse[idx] > threshold),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    cfg_dict = _load_config(args.config)

    ckpt_root = _resolve_path(REPO_ROOT, args.ckpt_root or cfg_dict.get("ckpt_root", "ckpt"))

    # Resolve DINOv3 source-repo path. Precedence:
    #   CLI flag > LMC_DINOV3_REPO env var > config file value.
    # The default config ships a placeholder ("/path/to/dinov3") so a fresh
    # user is forced to pick a real path explicitly.
    dino_repo_raw = (
        args.dino_repo_dir
        or os.environ.get("LMC_DINOV3_REPO")
        or cfg_dict.get("dino_repo_dir")
    )
    if dino_repo_raw is None or str(dino_repo_raw) in {"", "/path/to/dinov3"}:
        raise SystemExit(
            "ERROR: DINOv3 source-repo path is not set.\n"
            "  Provide it via one of (in order of precedence):\n"
            "    1. CLI flag        : --dino_repo_dir /path/to/your/dinov3\n"
            "    2. Environment var : export LMC_DINOV3_REPO=/path/to/your/dinov3\n"
            "    3. configs/inference_default.yaml -> dino_repo_dir\n"
            "  This must point to a local clone of https://github.com/facebookresearch/dinov3 ."
        )
    dino_repo = Path(dino_repo_raw)
    if not dino_repo.is_dir():
        raise SystemExit(
            f"ERROR: DINOv3 source-repo path does not exist or is not a directory: {dino_repo}\n"
            "  Clone https://github.com/facebookresearch/dinov3 and point LMC_DINOV3_REPO at it."
        )
    nnunet_lambda = float(args.nnunet_lambda if args.nnunet_lambda is not None else cfg_dict.get("nnunet_lambda", 0.77))
    folds = list(cfg_dict.get("folds", [0, 1, 2, 3, 4]))
    device = str(args.device or cfg_dict.get("device", "cuda"))
    amp = bool(cfg_dict.get("amp", True))

    cfg = InferenceConfig(
        ckpt_root=ckpt_root,
        dino_repo_dir=dino_repo,
        nnunet_lambda=nnunet_lambda,
        folds=tuple(folds),
        device=device,
        amp=amp,
    )

    mode = args.mode.strip()
    single_fold: Optional[int] = None
    if mode.startswith("fold:"):
        graph_branch_mode = "single"
        pixel_branch_mode = "single"
        single_fold = int(mode.split(":", 1)[1])
    else:
        graph_branch_mode = "ensemble"
        pixel_branch_mode = "ensemble"

    image_path = Path(args.image).resolve()
    caseid = image_path.stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = EndToEndInference(cfg)
    result = pipe.run_single_frame(
        image_path=image_path,
        caseid=caseid,
        graph_branch_mode=graph_branch_mode,
        pixel_branch_mode=pixel_branch_mode,
        single_fold=single_fold,
    )

    # Save vessel mask
    Image.fromarray(result["vessel_mask_uint8"]).save(out_dir / f"{caseid}_vessel_mask.png")

    # Save per-node predictions
    pred_records = _per_node_records(
        result["graph"],
        result["p_gnn"],
        result["p_nn"],
        result["p_fuse"],
        threshold=float(args.threshold),
    )
    pred_path = out_dir / f"{caseid}_pred.json"
    pred_path.write_text(json.dumps({
        "caseid": caseid,
        "image_path": str(image_path),
        "lambda": nnunet_lambda,
        "mode": mode,
        "single_fold": single_fold,
        "threshold": float(args.threshold),
        "num_nodes": int(result["num_nodes"]),
        "predictions": pred_records,
    }, indent=2))

    # Save the predicted graph topology (lets downstream tooling reuse it).
    (out_dir / f"{caseid}_graph_pred.json").write_text(json.dumps(result["graph"]))

    # Save overlay
    _draw_overlay(
        image_path=image_path,
        graph=result["graph"],
        p_score=np.asarray([rec["p_fuse"] for rec in pred_records], dtype=np.float32),
        threshold=float(args.threshold),
        out_path=out_dir / f"{caseid}_overlay.png",
    )

    print(json.dumps({
        "caseid": caseid,
        "num_nodes": int(result["num_nodes"]),
        "positive_at_threshold": int(sum(rec["hard_pred"] for rec in pred_records)),
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
