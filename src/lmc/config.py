"""Inference configuration for the LMC end-to-end pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class InferenceConfig:
    ckpt_root: Path
    dino_repo_dir: Path
    nnunet_lambda: float = 0.77
    folds: Sequence[int] = (0, 1, 2, 3, 4)
    device: str = "cuda"
    amp: bool = True

    @property
    def vessel_seg_dir(self) -> Path:
        return Path(self.ckpt_root) / "vessel_seg_nnunet"

    @property
    def pixel_branch_dir(self) -> Path:
        return Path(self.ckpt_root) / "pixel_branch_nnunet"

    @property
    def graph_branch_dir(self) -> Path:
        return Path(self.ckpt_root) / "graph_branch_gat"

    @property
    def dino_weights(self) -> Path:
        return Path(self.ckpt_root) / "dinov3" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
