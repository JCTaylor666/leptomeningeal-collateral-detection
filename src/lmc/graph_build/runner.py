"""Single-frame graph builder.

Wraps `lmc.preprocess.mask_to_graph` so callers can go from a binary
vessel mask (HxW uint8) to the same JSON record that the graph-branch
dataset loader expects, without touching the disk-bound CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .mask_to_graph import (
    build_eid2int,
    compute_node_features,
    connect_components,
    edge_adjacency,
    pixel_edge_assignment,
    skeletonize_2d,
    topology_extract,
)


def _binary_mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    bw = (mask > 0).astype(np.uint8) * 255
    return np.stack([bw, bw, bw], axis=-1)


def build_graph_from_mask(
    mask: np.ndarray,
    px: float = 1.0,
    py: float = 1.0,
    source_name: str = "frame.png",
) -> Optional[Dict]:
    """Convert a binary vessel mask into a graph JSON record.

    Returns a dict matching the on-disk `<caseid>_graph.json` schema:
    {nodes: [...], edges: [(u,v),...], edge_id_img_shape: [H,W],
     num_nodes, num_edges, source}.
    Returns None if the mask is empty or yields no nodes.
    """
    rgb = _binary_mask_to_rgb(mask)
    vessel_bw = (rgb[..., 0] > 0) | (rgb[..., 1] > 0) | (rgb[..., 2] > 0)
    if not vessel_bw.any():
        return None

    skel0 = skeletonize_2d(vessel_bw)
    G = topology_extract(skel0)
    edge_id_img = pixel_edge_assignment(vessel_bw, G, px, py)
    if edge_id_img.max() == 0:
        edge_id_img[vessel_bw] = 1

    stats = compute_node_features(edge_id_img, G, px=px, py=py)
    eid2int = build_eid2int(G)
    adj = edge_adjacency(G, eid2int)
    edge_ids = sorted(stats.keys())
    if not edge_ids:
        return None

    adj_map = {eid: set(adj.get(eid, [])) for eid in edge_ids}
    centers = {eid: stats[eid]["center"] for eid in edge_ids}

    for eid in edge_ids:
        if not adj_map[eid] and len(edge_ids) > 1:
            cy, cx = centers[eid]
            best = None
            best_d = 1e18
            for other in edge_ids:
                if other == eid:
                    continue
                oy, ox = centers[other]
                d = (cy - oy) ** 2 + (cx - ox) ** 2
                if d < best_d:
                    best_d = d
                    best = other
            if best is not None:
                adj_map[eid].add(best)
                adj_map[best].add(eid)

    adj_map = connect_components(adj_map, centers)

    edge_pairs: set = set()
    for u, neighs in adj_map.items():
        for v in neighs:
            if u == v:
                continue
            edge_pairs.add(tuple(sorted((u, v))))
    edge_list = sorted(edge_pairs)

    nodes: List[Dict] = []
    for eid in edge_ids:
        entry = dict(stats[eid])
        entry["id"] = int(eid)
        nodes.append(entry)

    record = {
        "source": source_name,
        "edge_id_img_shape": [int(edge_id_img.shape[0]), int(edge_id_img.shape[1])],
        "nodes": nodes,
        "edges": [list(pair) for pair in edge_list],
        "num_nodes": len(nodes),
        "num_edges": len(edge_list),
    }
    return record


def sort_nodes_by_id(nodes: Sequence[Dict]) -> List[Dict]:
    return sorted(nodes, key=lambda node: int(node.get("id", 0)))
