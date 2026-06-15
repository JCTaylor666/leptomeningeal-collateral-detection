"""λ-blend fusion of graph-branch and pixel-branch node probabilities.

p_fuse = λ · p_gnn + (1 − λ) · p_nn          (paper Eq. 3)
"""

from __future__ import annotations

import numpy as np


def fuse(p_gnn: np.ndarray, p_nn: np.ndarray, lam: float) -> np.ndarray:
    if not (0.0 <= float(lam) <= 1.0):
        raise ValueError(f"lambda must be in [0, 1], got {lam}")
    p_gnn = np.asarray(p_gnn, dtype=np.float32).reshape(-1)
    p_nn = np.asarray(p_nn, dtype=np.float32).reshape(-1)
    if p_gnn.shape != p_nn.shape:
        raise ValueError(f"shape mismatch: p_gnn={p_gnn.shape} p_nn={p_nn.shape}")
    return float(lam) * p_gnn + (1.0 - float(lam)) * p_nn
