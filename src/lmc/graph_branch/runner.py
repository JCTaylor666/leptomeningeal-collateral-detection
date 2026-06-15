"""Graph-branch runner — DINOv3 ViT-L/16 frozen + GAT, 5-fold ensemble.

Mirrors `EdgeGraphFeatureDataset._build_item` exactly so the per-node
sigmoid probabilities match the trained checkpoints byte-for-byte (up
to bf16 numerical noise).

Inputs per frame:
- image_float    : (H, W) in [0, 1]
- vessel_mask    : (H, W) in {0, 1}
- graph dict     : nodes [{id, bbox, center_proj, pixels}], edges
- x_morph (N, M) : delay prior morphology features (paper uses M=4).
                   When unavailable for a fresh frame, pass zeros — a
                   non-fatal degradation; the saved checkpoints still
                   produce reasonable output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .model import EdgeGraphFeatureNet


def _build_model_from_cfg(cfg: Dict, device: torch.device, dino_weights: str) -> EdgeGraphFeatureNet:
    return EdgeGraphFeatureNet(
        morph_dim=int(cfg.get("morph_dim", 4)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("gcn_layers", 2)),
        dyn_mlp_layers=int(cfg.get("dyn_mlp_layers", 2)),
        morph_mlp_layers=int(cfg.get("morph_mlp_layers", 2)),
        gnn_type=str(cfg.get("gnn_type", "gat")),
        ablate_roi=bool("no_roi" in str(cfg.get("ablate_list", ""))),
        ablate_center=bool("no_center" in str(cfg.get("ablate_list", ""))),
        ablate_pos=bool(cfg.get("ablate_pos", "no_pos" in str(cfg.get("ablate_list", "")))),
        use_pos_embed=bool(cfg.get("use_pos_embed", False)),
        use_vessel_mask=bool(cfg.get("use_vessel_mask", True)),
        roi_mode=str(cfg.get("roi_mode", "bbox")),
        roi_pool=str(cfg.get("roi_pool", "meanmax")),
        use_bbox_hw=bool(cfg.get("use_bbox_hw", True)),
        gat_num_heads=int(cfg.get("gat_num_heads", 1)),
        gat_feat_drop=float(cfg.get("gat_feat_drop", 0.0)),
        gat_attn_drop=float(cfg.get("gat_attn_drop", 0.2)),
        gat_residual=bool(cfg.get("gat_residual", True)),
        gin_aggregator_type=str(cfg.get("gin_aggregator_type", "sum")),
        gin_init_eps=float(cfg.get("gin_init_eps", 0.0)),
        gin_learn_eps=bool(cfg.get("gin_learn_eps", True)),
        layer_concat_enable=bool(cfg.get("layer_concat_enable", False)),
        backbone_type=str(cfg.get("backbone_type", "dino")),
        dino_model_name=str(cfg.get("dino_model_name", "dinov3_vitl16")),
        dino_repo_dir=str(cfg.get("dino_repo_dir", "")),
        dino_backbone_weights=str(dino_weights),
        dino_freeze=bool(cfg.get("dino_freeze", True)),
        dino_use_autocast_bf16=bool(cfg.get("dino_use_autocast_bf16", True)),
        dino_gray_to_rgb=bool(cfg.get("dino_gray_to_rgb", True)),
        dino_layer_select=int(cfg.get("dino_layer_select", -1)),
        pos_freq=int(cfg.get("pos_freq", 6)),
    ).to(device)


class GraphBranchRunner:
    def __init__(
        self,
        ckpt_dir: Path,
        dino_repo_dir: Path,
        dino_weights: Path,
        device: Optional[torch.device] = None,
        amp: bool = True,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.dino_repo_dir = Path(dino_repo_dir)
        self.dino_weights = Path(dino_weights)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(amp)

        cfg_path = self.ckpt_dir / "train_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing graph-branch train_config.json: {cfg_path}")
        cfg = json.loads(cfg_path.read_text())
        cfg["dino_repo_dir"] = str(self.dino_repo_dir)
        self.cfg = cfg
        self.morph_dim = int(cfg.get("morph_dim", 4))

        self._models_by_fold: Dict[int, EdgeGraphFeatureNet] = {}

    def _load_fold(self, fold: int) -> EdgeGraphFeatureNet:
        if fold in self._models_by_fold:
            return self._models_by_fold[fold]
        ckpt_path = self.ckpt_dir / f"fold{int(fold)}_best_prauc.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing graph-branch checkpoint: {ckpt_path}")
        model = _build_model_from_cfg(self.cfg, self.device, str(self.dino_weights))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt)
        load_result = model.load_state_dict(state, strict=False)
        missing = [k for k in load_result.missing_keys if not k.startswith("backbone.dino.")]
        unexpected = [k for k in load_result.unexpected_keys if not k.startswith("backbone.dino.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load mismatch for {ckpt_path}. "
                f"missing={len(missing)}({missing[:6]}) unexpected={len(unexpected)}({unexpected[:6]})"
            )
        model.eval()
        self._models_by_fold[fold] = model
        return model

    @staticmethod
    def _resize_to(arr: np.ndarray, target_hw: Sequence[int], nearest: bool = False) -> np.ndarray:
        from PIL import Image as _Image

        if arr.shape == tuple(target_hw):
            return arr
        if nearest:
            tmp = _Image.fromarray((arr * 255).astype(np.uint8))
            tmp = tmp.resize((int(target_hw[1]), int(target_hw[0])), resample=_Image.NEAREST)
            return (np.asarray(tmp, dtype=np.uint8) > 0).astype(np.float32)
        tmp = _Image.fromarray((arr * 255).astype(np.uint8))
        tmp = tmp.resize((int(target_hw[1]), int(target_hw[0])), resample=_Image.BILINEAR)
        return np.asarray(tmp, dtype=np.float32) / 255.0

    def build_inputs(
        self,
        graph: Dict,
        image: np.ndarray,
        vessel_mask: np.ndarray,
        x_morph: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        nodes = sorted(graph.get("nodes", []), key=lambda node: int(node["id"]))
        if not nodes:
            raise ValueError("Graph has no nodes")

        edge_ids = [int(node["id"]) for node in nodes]
        edge_id_to_idx = {eid: idx for idx, eid in enumerate(edge_ids)}
        bbox_arr = np.asarray([tuple(node["bbox"]) for node in nodes], dtype=np.float32)

        centers_nn: List[List[float]] = []
        for node in nodes:
            cproj = node.get("center_proj", None)
            if not (isinstance(cproj, (list, tuple)) and len(cproj) >= 2):
                raise ValueError(f"node {node.get('id')}: missing center_proj")
            centers_nn.append([float(cproj[0]), float(cproj[1])])
        centers_arr = np.asarray(centers_nn, dtype=np.float32)

        adj = np.zeros((len(edge_ids), len(edge_ids)), dtype=np.float32)
        edge_pairs: List[Tuple[int, int]] = []
        for u, v in graph.get("edges", []):
            ui, vi = int(u), int(v)
            if ui in edge_id_to_idx and vi in edge_id_to_idx:
                i = edge_id_to_idx[ui]
                j = edge_id_to_idx[vi]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                edge_pairs.append((a, b))

        if "edge_id_img_shape" in graph:
            target_hw = tuple(graph["edge_id_img_shape"])
        else:
            target_hw = image.shape

        if image.shape != target_hw:
            image = self._resize_to(image, target_hw, nearest=False)
        if vessel_mask.shape != target_hw:
            vessel_mask = self._resize_to(vessel_mask, target_hw, nearest=True)

        # x_morph: shape (N, morph_dim). If not provided, zeros.
        if x_morph is None:
            x_morph_arr = np.zeros((len(edge_ids), max(0, self.morph_dim)), dtype=np.float32)
        else:
            x_morph_arr = np.asarray(x_morph, dtype=np.float32)
            if x_morph_arr.shape != (len(edge_ids), self.morph_dim):
                raise ValueError(
                    f"x_morph shape {x_morph_arr.shape} != expected ({len(edge_ids)}, {self.morph_dim})"
                )

        pixels: List[torch.Tensor] = []
        for node in nodes:
            pix = node.get("pixels", [])
            if isinstance(pix, list) and pix:
                pixels.append(torch.tensor(pix, dtype=torch.long))
            else:
                pixels.append(torch.zeros((0, 2), dtype=torch.long))

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long)
        else:
            edge_index = torch.empty((0, 2), dtype=torch.long)

        frames = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        vessel = torch.from_numpy(vessel_mask).unsqueeze(0).unsqueeze(0)
        return {
            "frames": frames,
            "vessel_masks": vessel,
            "bboxes": torch.from_numpy(bbox_arr),
            "centers": torch.from_numpy(centers_arr),
            "x_morph": torch.from_numpy(x_morph_arr),
            "adj": torch.from_numpy(adj),
            "edge_index": edge_index,
            "pixels": pixels,
        }, edge_ids

    @torch.no_grad()
    def forward_one_frame(
        self,
        graph: Dict,
        image: np.ndarray,
        vessel_mask: np.ndarray,
        fold: int,
        x_morph: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        model = self._load_fold(int(fold))
        inputs, _ = self.build_inputs(graph, image, vessel_mask, x_morph=x_morph)
        device = self.device
        amp_enable = bool(self.amp and device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=amp_enable):
            logits = model(
                inputs["frames"].to(device),
                inputs["bboxes"].to(device),
                inputs["centers"].to(device),
                inputs["x_morph"].to(device),
                inputs["adj"].to(device),
                inputs["edge_index"].to(device),
                vessel_masks=inputs["vessel_masks"].to(device),
                pixels=[p.to(device) for p in inputs["pixels"]],
            )
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).reshape(-1)

    @torch.no_grad()
    def forward_one_frame_ensemble(
        self,
        graph: Dict,
        image: np.ndarray,
        vessel_mask: np.ndarray,
        folds: Sequence[int] = (0, 1, 2, 3, 4),
        x_morph: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not folds:
            raise ValueError("ensemble must use at least 1 fold")
        accum: Optional[np.ndarray] = None
        for fold in folds:
            prob = self.forward_one_frame(graph, image, vessel_mask, fold=int(fold), x_morph=x_morph)
            accum = prob if accum is None else accum + prob
        return (accum / float(len(folds))).astype(np.float32)


def load_prior_features(prior_npz_path, edge_ids: Sequence[int], feature_dim: int = 4) -> np.ndarray:
    """Load delay-prior morphology features and align to graph node order."""
    prior_npz_path = Path(prior_npz_path)
    if not prior_npz_path.exists():
        return np.zeros((len(edge_ids), feature_dim), dtype=np.float32)
    with np.load(prior_npz_path) as data:
        prior_eids = np.asarray(data["edge_ids"], dtype=np.int32)
        prior_x = np.asarray(data["x_morph"], dtype=np.float32)
    if prior_x.shape[1] != feature_dim:
        return np.zeros((len(edge_ids), feature_dim), dtype=np.float32)
    eid_to_row = {int(eid): i for i, eid in enumerate(prior_eids.tolist())}
    out = np.zeros((len(edge_ids), feature_dim), dtype=np.float32)
    for idx, eid in enumerate(edge_ids):
        if int(eid) in eid_to_row:
            out[idx] = prior_x[eid_to_row[int(eid)]]
    return out
