"""Rendering helpers for the LMC inference web demo.

Each function turns one pipeline artifact (vessel mask, graph, pixel
probability map, per-node scores) into an inline base64 PNG ("data:" URI)
that the browser can drop straight into an <img>. No matplotlib — a small
hand-rolled colormap keeps the demo dependency footprint to Pillow + numpy
(already required by the inference stack).
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

# Turbo-ish perceptual ramp (low → high). Used for probability heatmaps so
# weak and strong responses stay visually distinct.
_ANCHOR_T = np.array([0.00, 0.13, 0.25, 0.38, 0.50, 0.63, 0.75, 0.88, 1.00])
_ANCHOR_C = np.array(
    [
        [48, 18, 59],
        [70, 107, 227],
        [40, 170, 255],
        [50, 220, 200],
        [120, 250, 120],
        [220, 240, 60],
        [255, 180, 40],
        [240, 90, 30],
        [150, 20, 10],
    ],
    dtype=np.float64,
)


def colormap(t: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB uint8 via the turbo-ish ramp."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    out = np.empty(t.shape + (3,), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(t, _ANCHOR_T, _ANCHOR_C[:, c])
    return out.astype(np.uint8)


def _gray_rgb(image_uint8: np.ndarray) -> np.ndarray:
    """(H, W) grayscale → (H, W, 3) uint8."""
    g = np.asarray(image_uint8)
    if g.ndim == 3:
        g = g[..., 0]
    g = g.astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _encode(arr_or_img) -> str:
    if isinstance(arr_or_img, np.ndarray):
        img = Image.fromarray(arr_or_img.astype(np.uint8))
    else:
        img = arr_or_img
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_image(image_uint8: np.ndarray) -> str:
    """Encode the raw input frame for echo-back."""
    return _encode(_gray_rgb(image_uint8))


def vessel_overlay(image_uint8: np.ndarray, mask: np.ndarray) -> str:
    """DSA frame with the predicted vessel mask tinted green."""
    base = _gray_rgb(image_uint8).astype(np.float64)
    m = (np.asarray(mask) > 0)
    tint = np.array([40, 230, 120], dtype=np.float64)
    base[m] = 0.35 * base[m] + 0.65 * tint
    return _encode(np.clip(base, 0, 255).astype(np.uint8))


def masked_input(image_uint8: np.ndarray, mask: np.ndarray) -> str:
    """image * (mask > 0) — exactly what the pixel branch ingests."""
    g = np.asarray(image_uint8)
    if g.ndim == 3:
        g = g[..., 0]
    m = (np.asarray(mask) > 0).astype(np.uint8)
    masked = (g.astype(np.uint8) * m).astype(np.uint8)
    return _encode(_gray_rgb(masked))


def graph_overlay(image_uint8: np.ndarray, graph: Dict) -> str:
    """Draw the vessel-segment graph (edges + node markers) over the frame."""
    base = Image.fromarray(_gray_rgb(image_uint8)).convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")
    nodes = sorted(graph.get("nodes", []), key=lambda n: int(n["id"]))
    centers: Dict[int, tuple] = {}
    for node in nodes:
        y0, y1, x0, x1 = (int(v) for v in node["bbox"])
        centers[int(node["id"])] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    # Edges first (under the node markers).
    for u, v in graph.get("edges", []):
        if int(u) in centers and int(v) in centers:
            draw.line([centers[int(u)], centers[int(v)]], fill=(90, 200, 255, 150), width=1)
    # Node markers.
    for cx, cy in centers.values():
        draw.ellipse([cx - 1.6, cy - 1.6, cx + 1.6, cy + 1.6], fill=(255, 210, 60, 230))
    return _encode(base)


def pixel_heatmap(image_uint8: np.ndarray, prob_map: np.ndarray) -> str:
    """Dense pixel-branch probability as a heatmap blended over the frame."""
    base = _gray_rgb(image_uint8).astype(np.float64)
    p = np.clip(np.asarray(prob_map, dtype=np.float64), 0.0, 1.0)
    heat = colormap(p).astype(np.float64)
    # Alpha grows with probability so the (masked) background stays dark.
    a = (p ** 0.6)[..., None]
    out = base * (1.0 - a) + heat * a
    return _encode(np.clip(out, 0, 255).astype(np.uint8))


def node_score_overlay(
    image_uint8: np.ndarray,
    nodes: Sequence[Dict],
    scores: np.ndarray,
    alpha: float = 0.85,
) -> str:
    """Colour each node's pixels by its (continuous) score via the colormap."""
    base = _gray_rgb(image_uint8).astype(np.float64)
    h, w = base.shape[:2]
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    colours = colormap(scores).astype(np.float64)  # (N, 3)
    for idx, node in enumerate(nodes):
        pix = node.get("pixels", [])
        if not pix:
            continue
        arr = np.asarray(pix, dtype=np.int64)
        ys = np.clip(arr[:, 0], 0, h - 1)
        xs = np.clip(arr[:, 1], 0, w - 1)
        col = colours[idx]
        base[ys, xs] = (1.0 - alpha) * base[ys, xs] + alpha * col
    return _encode(np.clip(base, 0, 255).astype(np.uint8))


def final_overlay(
    image_uint8: np.ndarray,
    nodes: Sequence[Dict],
    scores: np.ndarray,
    threshold: float,
) -> str:
    """Decision view: positive nodes filled red + boxed; others faint cyan box.

    Matches scripts/infer_one.py `_draw_overlay` so the demo's final frame
    is identical to the CLI's `<caseid>_overlay.png`.
    """
    base = Image.fromarray(_gray_rgb(image_uint8)).convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    for idx, node in enumerate(nodes):
        y0, y1, x0, x1 = (int(v) for v in node["bbox"])
        if scores[idx] >= threshold:
            for px in node.get("pixels", []):
                draw.point((int(px[1]), int(px[0])), fill=(255, 70, 70, 220))
            draw.rectangle([x0, y0, x1, y1], outline=(255, 70, 70, 255), width=1)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=(70, 200, 220, 120), width=1)
    return _encode(base)


def colorbar(width: int = 160, height: int = 12) -> str:
    """A horizontal 0→1 colorbar legend matching `colormap`."""
    ramp = np.linspace(0.0, 1.0, width)
    row = colormap(ramp)  # (width, 3)
    img = np.repeat(row[None, :, :], height, axis=0)
    return _encode(img.astype(np.uint8))
