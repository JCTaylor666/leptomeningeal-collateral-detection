"""End-to-end inference orchestrator.

Pipeline (paper Sec. 4):
  raw DSA frame
    └─ vessel_seg_runner   ── nnU-Net + clDice 5-fold ensemble  → vessel mask
        └─ graph_runner    ── line-graph from skeleton          → graph JSON
            ├─ graph_branch_runner   ── DINOv3 + GAT 5-fold      → per-node p_gnn
            └─ pixel_branch_runner   ── nnU-Net 5-fold + Eq. 2   → per-node p_nn
                └─ fusion.fuse  (λ-blend Eq. 3)                 → per-node p_fuse
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .config import InferenceConfig
from .fusion import fuse
from .graph_branch.runner import GraphBranchRunner
from .graph_build import build_graph_from_mask, sort_nodes_by_id
from .pixel_branch.runner import PixelBranchRunner, pool_pixel_prob_to_nodes
from .vessel_seg.runner import VesselSegRunner


class EndToEndInference:
    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self._vessel_seg: Optional[VesselSegRunner] = None
        self._pixel_branch: Optional[PixelBranchRunner] = None
        self._graph_branch: Optional[GraphBranchRunner] = None

    @property
    def vessel_seg(self) -> VesselSegRunner:
        if self._vessel_seg is None:
            self._vessel_seg = VesselSegRunner(model_dir=self.cfg.vessel_seg_dir, device=self.device)
        return self._vessel_seg

    @property
    def pixel_branch(self) -> PixelBranchRunner:
        if self._pixel_branch is None:
            self._pixel_branch = PixelBranchRunner(model_dir=self.cfg.pixel_branch_dir, device=self.device)
        return self._pixel_branch

    @property
    def graph_branch(self) -> GraphBranchRunner:
        if self._graph_branch is None:
            self._graph_branch = GraphBranchRunner(
                ckpt_dir=self.cfg.graph_branch_dir,
                dino_repo_dir=self.cfg.dino_repo_dir,
                dino_weights=self.cfg.dino_weights,
                device=self.device,
                amp=self.cfg.amp,
            )
        return self._graph_branch

    @staticmethod
    def _load_image(image_path: Path) -> np.ndarray:
        return np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)

    def run_single_frame(
        self,
        image_path: Path,
        caseid: str,
        precomputed_graph: Optional[Dict] = None,
        precomputed_vessel_mask: Optional[np.ndarray] = None,
        graph_branch_mode: str = "ensemble",
        pixel_branch_mode: str = "ensemble",
        single_fold: Optional[int] = None,
        x_morph: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run full pipeline on one frame and return a structured result.

        graph_branch_mode / pixel_branch_mode ∈ {"ensemble", "single"}.
        When mode is "single", `single_fold` selects the fold.
        """
        image_path = Path(image_path)
        image_uint8 = self._load_image(image_path)
        image_float = image_uint8.astype(np.float32) / 255.0

        if precomputed_vessel_mask is not None:
            vessel_mask_uint8 = (np.asarray(precomputed_vessel_mask) > 0).astype(np.uint8) * 255
        else:
            vessel_mask_uint8, _ = self.vessel_seg.predict_mask(image_uint8, caseid=caseid, return_probability=False)
        vessel_mask_float = (vessel_mask_uint8 > 0).astype(np.float32)

        if precomputed_graph is not None:
            graph = precomputed_graph
        else:
            graph = build_graph_from_mask(vessel_mask_uint8, source_name=image_path.name)
            if graph is None:
                raise RuntimeError(f"Failed to build graph for {caseid} (empty vessel mask?)")

        nodes = sort_nodes_by_id(graph.get("nodes", []))
        graph_for_branch = {**graph, "nodes": nodes}

        if pixel_branch_mode == "single":
            if single_fold is None:
                raise ValueError("pixel_branch_mode='single' needs single_fold")
            pixel_prob = self.pixel_branch.predict_prob(
                image_uint8, caseid=caseid, folds=[int(single_fold)], vessel_mask=vessel_mask_uint8
            )
        else:
            pixel_prob = self.pixel_branch.predict_prob(
                image_uint8, caseid=caseid, folds=tuple(self.cfg.folds), vessel_mask=vessel_mask_uint8
            )

        target_hw = tuple(graph_for_branch.get("edge_id_img_shape", image_float.shape))
        if pixel_prob.shape != target_hw:
            from PIL import Image as _Image

            pp = _Image.fromarray((pixel_prob * 65535).astype(np.uint16))
            pp = pp.resize((int(target_hw[1]), int(target_hw[0])), resample=_Image.BILINEAR)
            pixel_prob = np.asarray(pp, dtype=np.float32) / 65535.0

        p_nn = pool_pixel_prob_to_nodes(pixel_prob, nodes)

        if graph_branch_mode == "single":
            if single_fold is None:
                raise ValueError("graph_branch_mode='single' needs single_fold")
            p_gnn = self.graph_branch.forward_one_frame(
                graph_for_branch,
                image=image_float,
                vessel_mask=vessel_mask_float,
                fold=int(single_fold),
                x_morph=x_morph,
            )
        else:
            p_gnn = self.graph_branch.forward_one_frame_ensemble(
                graph_for_branch,
                image=image_float,
                vessel_mask=vessel_mask_float,
                folds=tuple(self.cfg.folds),
                x_morph=x_morph,
            )

        p_fuse = fuse(p_gnn, p_nn, lam=float(self.cfg.nnunet_lambda))

        return {
            "caseid": caseid,
            "image_path": str(image_path),
            "num_nodes": int(len(nodes)),
            "lambda": float(self.cfg.nnunet_lambda),
            "graph_branch_mode": graph_branch_mode,
            "pixel_branch_mode": pixel_branch_mode,
            "single_fold": int(single_fold) if single_fold is not None else None,
            "graph": graph,
            "vessel_mask_uint8": vessel_mask_uint8,
            "pixel_prob_map": pixel_prob,
            "node_ids": [int(node["id"]) for node in nodes],
            "node_bboxes": [list(map(int, node["bbox"])) for node in nodes],
            "p_gnn": p_gnn,
            "p_nn": p_nn,
            "p_fuse": p_fuse,
        }
