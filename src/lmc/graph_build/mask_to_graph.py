#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import color, filters, measure, morphology, util
from skimage.color import rgba2rgb
from skimage.morphology import skeletonize
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
'''
python /home/juncao/data/LMC/STGRAPH/Preprocess/vessel2d_pipeline.py \
  --input_dir /home/juncao/data/big_storage/jyc/data/Zurich_DSA_Final/mask_unify \
  --collateral_dir /home/juncao/data/big_storage/jyc/data/Zurich_DSA_Final/collateral_mask_unify \
  --json_dir /home/juncao/data/big_storage/jyc/data/ZURICH_GRAPH/GRAPH_JSON \
  --output_dir /home/juncao/data/big_storage/jyc/data/ZURICH_GRAPH/GRAPH_VIS \
  --pattern "s*_f*.png" \
  --collateral_ratio 0.1 \
  --num_workers 16




'''
def visualize_edge_id_img(edge_id_img, out_png="edge_id_img.png"):
    """
    可视化 edge_id_img:
      - 背景(0) = 黑色
      - 每条边 = 随机颜色
    """
    edge_ids = np.unique(edge_id_img)
    edge_ids = edge_ids[edge_ids > 0]  # 去掉背景0

    # 为每个边分配一个随机颜色
    rng = np.random.default_rng(42)
    colors = rng.uniform(0.2, 1.0, size=(len(edge_ids), 3))  # 避免太暗
    cmap_colors = np.vstack(([0, 0, 0], colors))  # 0=黑色
    cmap = ListedColormap(cmap_colors)

    # 建立一个映射，把edge_id映射到连续索引
    id2idx = {eid: i+1 for i, eid in enumerate(edge_ids)}
    img_vis = np.zeros_like(edge_id_img, dtype=int)
    for eid, idx in id2idx.items():
        img_vis[edge_id_img == eid] = idx

    plt.figure(figsize=(6,6))
    plt.imshow(img_vis, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[可视化结果保存到] {out_png}")


def neighbors8(r, c, shape):
    rr = [r-1,r-1,r-1, r, r, r+1,r+1,r+1]
    cc = [c-1,c,  c+1, c-1,c+1, c-1,c, c+1]
    out = []
    for y,x in zip(rr,cc):
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            out.append((y,x))
    return out

def degree_img(skel):
    H,W = skel.shape
    deg = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.nonzero(skel)
    S = set(zip(ys,xs))
    for y,x in zip(ys,xs):
        d = 0
        for ny,nx in neighbors8(y,x,skel.shape):
            if (ny,nx) in S:
                d += 1
        deg[y,x] = d
    return deg

def load_rgb_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def is_max_mask(path: Path) -> bool:
    stem = path.stem.lower()
    return ("_max" in stem) and ("minip" not in stem)

def skeletonize_2d(bw):
    skel = skeletonize(bw)
    skel = morphology.thin(skel)
    return skel

def topology_extract(skel):
    deg = degree_img(skel)
    sk_coords = np.column_stack(np.nonzero(skel))
    node_mask = np.zeros_like(skel, dtype=bool)
    node_mask[deg != 2] = skel[deg != 2]
    node_lab = measure.label(node_mask, connectivity=2)
    node_props = measure.regionprops(node_lab)
    G = nx.Graph()
    node_label_to_id = {}
    for i, prop in enumerate(node_props):
        cy, cx = prop.coords.mean(axis=0)
        nid = f"N{i}"
        node_label_to_id[prop.label] = nid
        G.add_node(nid, y=float(cy), x=float(cx), pixels=[(int(y),int(x)) for y,x in prop.coords])
    skel_set = set(map(tuple, sk_coords))
    visited = set()
    edges = []
    for region in node_props:
        for (y,x) in region.coords:
            for ny,cx in neighbors8(y,x,skel.shape):
                if (ny,cx) in skel_set and node_lab[ny,cx] == 0 and (ny,cx) not in visited:
                    path = [(y,x), (ny,cx)]
                    prev = (y,x)
                    cur = (ny,cx)
                    visited.add(cur)
                    while True:
                        next_pixels = []
                        for py,px in neighbors8(cur[0],cur[1],skel.shape):
                            if (py,px) in skel_set and (py,px) != prev:
                                next_pixels.append((py,px))
                        if len(next_pixels) == 0:
                            break
                        if len(next_pixels) > 1 or node_lab[cur] != 0:
                            break
                        nxt = next_pixels[0]
                        path.append(nxt)
                        prev, cur = cur, nxt
                        visited.add(cur)
                        if node_lab[cur] != 0:
                            break
                    start_node_lab = node_lab[path[0]]
                    end_node_lab = node_lab[cur]
                    if start_node_lab == 0 or end_node_lab == 0:
                        continue
                    u = node_label_to_id[start_node_lab]
                    v = node_label_to_id[end_node_lab]
                    if u == v: continue
                    edges.append((u, v, {"centerline": [(int(py),int(px)) for py,px in path]}))
    for u,v,attr in edges:
        if G.has_edge(u,v):
            G[u][v]["centerline"] += attr["centerline"]
        else:
            G.add_edge(u,v, **attr)
    for i,(u,v,data) in enumerate(G.edges(data=True)):
        data["eid"] = f"E{i}"
    return G

def centerline_points_with_edge_ids(G):
    pts = []
    eids = []
    for u,v,data in G.edges(data=True):
        for (y,x) in data["centerline"]:
            pts.append((y,x)); eids.append(data["eid"])
    if len(pts)==0:
        return np.zeros((0,2), int), []
    return np.array(pts, dtype=int), eids

def build_eid2int(G):
    _, cl_eids = centerline_points_with_edge_ids(G)
    uniq = list(dict.fromkeys(cl_eids))
    return {e:i+1 for i,e in enumerate(uniq)}

def edge_adjacency(G, eid2int):
    adj = {}
    for n in G.nodes():
        inc = []
        for u,v,data in G.edges(n, data=True):
            eid = data.get("eid")
            if eid in eid2int:
                inc.append(eid2int[eid])
        for i in inc:
            adj.setdefault(i, set())
            for j in inc:
                if i != j:
                    adj[i].add(j)
    return {k: sorted(list(v)) for k, v in adj.items()}

def edge_stats(edge_id_img, include_pixels: bool = False):
    stats = {}
    edge_ids = [int(e) for e in np.unique(edge_id_img) if e > 0]
    for eid in edge_ids:
        ys, xs = np.where(edge_id_img == eid)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        cy = float(ys.mean())
        cx = float(xs.mean())
        entry = {"bbox": [y0, y1, x0, x1], "center": [cy, cx]}
        if include_pixels:
            entry["pixels"] = [[int(y), int(x)] for y, x in zip(ys.tolist(), xs.tolist())]
        stats[eid] = entry
    return stats


def compute_node_features(edge_id_img, G, px: float = 1.0, py: float = 1.0):
    stats = edge_stats(edge_id_img, include_pixels=True)
    if not stats:
        return stats

    eid2int = build_eid2int(G)
    int2edge = {}
    for u, v, data in G.edges(data=True):
        eid = data.get("eid")
        if eid in eid2int:
            int_id = eid2int[eid]
            int2edge[int_id] = (u, v, data.get("centerline", []))
    adj = edge_adjacency(G, eid2int)

    props = {prop.label: prop for prop in measure.regionprops(edge_id_img)}
    for eid, entry in stats.items():
        prop = props.get(eid)
        if prop is None:
            continue
        y0, x0, y1, x1 = prop.bbox
        bbox_h = int(y1 - y0)
        bbox_w = int(x1 - x0)
        bbox_area = float(bbox_h * bbox_w)
        entry["bbox_hw"] = [bbox_h, bbox_w]
        entry["bbox_area"] = bbox_area
        entry["bbox_aspect"] = float(bbox_w / bbox_h) if bbox_h > 0 else 0.0
        entry["center_bbox"] = [float(y0 + bbox_h / 2.0), float(x0 + bbox_w / 2.0)]

        u, v, cl = int2edge.get(eid, (None, None, []))
        if cl:
            cl_arr = np.array(cl, dtype=float)
            entry["center_skel"] = [float(cl_arr[:, 0].mean()), float(cl_arr[:, 1].mean())]
            # nearest centerline point to region centroid
            c = np.array(entry["center"], dtype=float)
            d = ((cl_arr - c) ** 2).sum(axis=1)
            nn = cl_arr[int(d.argmin())]
            entry["center_proj"] = [float(nn[0]), float(nn[1])]
            # orientation from end-to-end direction of centerline
            if len(cl_arr) >= 2:
                dy = (cl_arr[-1, 0] - cl_arr[0, 0]) * py
                dx = (cl_arr[-1, 1] - cl_arr[0, 1]) * px
                entry["orientation_rad"] = float(np.arctan2(dy, dx))
            else:
                entry["orientation_rad"] = 0.0
        else:
            entry["center_skel"] = entry["center"]
            entry["center_proj"] = entry["center"]
            entry["orientation_rad"] = 0.0
    return stats

def connected_components(adj_map):
    visited = set()
    comps = []
    for node in adj_map.keys():
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj_map.get(u, set()):
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps

def connect_components(adj_map, centers):
    comps = connected_components(adj_map)
    if len(comps) <= 1:
        return adj_map
    while len(comps) > 1:
        best_pair = None
        best_d = 1e18
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                for u in comps[i]:
                    cuy, cux = centers[u]
                    for v in comps[j]:
                        cvy, cvx = centers[v]
                        d = (cuy - cvy) ** 2 + (cux - cvx) ** 2
                        if d < best_d:
                            best_d = d
                            best_pair = (u, v)
        if best_pair is None:
            break
        u, v = best_pair
        adj_map[u].add(v)
        adj_map[v].add(u)
        comps = connected_components(adj_map)
    return adj_map

def pixel_edge_assignment(bw, G, px=1.0, py=1.0):
    H,W = bw.shape
    ys, xs = np.nonzero(bw)
    fg = np.column_stack([ys,xs])
    cl_pts, cl_eids = centerline_points_with_edge_ids(G)
    edge_id_img = np.zeros(bw.shape, dtype=np.int32)
    if len(cl_pts)==0 or len(fg)==0:
        return edge_id_img
    cl_phys = np.column_stack([cl_pts[:,0]*py, cl_pts[:,1]*px])
    tree = cKDTree(cl_phys)
    d, idx = tree.query(np.column_stack([fg[:,0]*py, fg[:,1]*px]), k=1)
    # map eid to int label
    uniq = list(dict.fromkeys(cl_eids))
    eid2int = {e:i+1 for i,e in enumerate(uniq)}
    assigned = np.array([eid2int[cl_eids[i]] for i in idx], dtype=np.int32)
    edge_id_img[fg[:,0], fg[:,1]] = assigned
    # simple unlabeled fill
    unlabeled = (edge_id_img==0) & bw
    iters=0
    while unlabeled.any() and iters<100:
        iters += 1
        changed=False
        ys, xs = np.nonzero(unlabeled)
        for y,x in zip(ys,xs):
            neigh = edge_id_img[max(0,y-1):y+2, max(0,x-1):x+2]
            vals, cnts = np.unique(neigh[neigh>0], return_counts=True)
            if len(vals)>0:
                edge_id_img[y,x] = int(vals[np.argmax(cnts)]); unlabeled[y,x]=False; changed=True
        if not changed: break
    return edge_id_img

def polyline_length(poly, px=1.0, py=1.0):
    if len(poly) < 2: return 0.0
    p = np.array(poly, dtype=float)
    dy = np.diff(p[:,0]) * py
    dx = np.diff(p[:,1]) * px
    return float(np.hypot(dy, dx).sum())

def feature_annotation(bw, G, edge_id_img, px=1.0, py=1.0):
    edt = ndi.distance_transform_edt(bw, sampling=(py,px))
    rows = []
    for u,v,data in G.edges(data=True):
        eid = data["eid"]
        # mask for this edge (based on nearest assignment)
        lbl = None
        # find a label of any centerline pixel
        for (y,x) in data["centerline"]:
            lbl = edge_id_img[y,x]
            if lbl>0: break
        mask = (edge_id_img==lbl) if (lbl is not None and lbl>0) else np.zeros_like(bw, bool)
        area = float(mask.sum()) * (px*py)
        length = polyline_length(data["centerline"], px, py)
        uy,ux = G.nodes[u]["y"], G.nodes[u]["x"]
        vy,vx = G.nodes[v]["y"], G.nodes[v]["x"]
        dist = math.hypot((uy-vy)*py, (ux-vx)*px)
        straight = (dist/length) if length>0 else 0.0
        sk_edge = np.zeros_like(bw, dtype=bool)
        for (y,x) in data["centerline"]:
            sk_edge[y,x] = True
        hw = edt[sk_edge]
        if hw.size==0:
            hw_min=hw_max=hw_avg=hw_std=0.0
        else:
            hw_min=float(np.min(hw)); hw_max=float(np.max(hw)); hw_avg=float(np.mean(hw)); hw_std=float(np.std(hw))
        roundness = (hw_max/hw_min) if hw_min>0 else np.inf
        avg_thickness = 2.0*hw_avg
        avg_cross = (area/length) if length>0 else 0.0
        # tip & inner_length (simplified approximations)
        deg_u = G.degree[u]; deg_v = G.degree[v]
        leaf = u if deg_u==1 else (v if deg_v==1 else None)
        tip_halfwidth = hw_min
        inner_length = 0.0
        if leaf is not None:
            ly,lx = G.nodes[leaf]["y"], G.nodes[leaf]["x"]
            cl = np.array(data["centerline"], dtype=float)
            d2 = ((cl[:,0]-ly)*py)**2 + ((cl[:,1]-lx)*px)**2
            tip_idx = int(np.argmin(d2))
            tip_halfwidth = float(edt[tuple(map(int, cl[tip_idx]))])
            branch = v if leaf==u else u
            by,bx = G.nodes[branch]["y"], G.nodes[branch]["x"]
            d_branch = np.hypot((cl[:,0]-by)*py,(cl[:,1]-bx)*px)
            inner_mask = d_branch <= (3.0*min(px,py))
            inner_length = polyline_length(cl[inner_mask].tolist(), px, py) if inner_mask.any() else 0.0
        avgHalfwidthMean = hw_avg if hw_avg>0 else 1e-6
        bulge_size = ( (length - inner_length + (tip_halfwidth or 0.0)) / avgHalfwidthMean ) if avgHalfwidthMean>0 else np.inf
        rows.append(dict(edge_id=eid, u=u, v=v, length=length, distance=dist, straightness=straight,
                         area=area, avg_halfwidth=hw_avg, min_halfwidth=hw_min, max_halfwidth=hw_max,
                         std_halfwidth=hw_std, roundness=roundness, avg_thickness=avg_thickness,
                         avg_cross_section_like=avg_cross, tip_halfwidth=tip_halfwidth,
                         inner_length=inner_length, bulge_size=bulge_size, deg_u=int(deg_u), deg_v=int(deg_v)))
    return pd.DataFrame(rows)

def refinement(G, df, bulge_thresh=2.0):
    G2 = G.copy()
    to_delete=set()
    for n in list(G2.nodes()):
        inc = list(G2.edges(n, data=True))
        if len(inc) <= 2: continue
        cand=[]
        for u,v,data in inc:
            row = df[df.edge_id==data["eid"]]
            if row.empty: continue
            row = row.iloc[0]
            if row.deg_u==1 or row.deg_v==1:
                if row.bulge_size < bulge_thresh:
                    cand.append((data["eid"], row.bulge_size))
        cand.sort(key=lambda t:t[1])
        removable=set([e for e,_ in cand])
        while len(inc) - len(removable) < 2 and removable:
            biggest=max(removable, key=lambda e:[bs for e2,bs in cand if e2==e][0])
            removable.remove(biggest)
        to_delete |= removable
    if not to_delete:
        return G2, 0
    for u,v,data in list(G2.edges(data=True)):
        if data["eid"] in to_delete:
            G2.remove_edge(u,v)
    iso=[n for n in G2.nodes() if G2.degree[n]==0]
    G2.remove_nodes_from(iso)
    return G2, len(to_delete)

def draw_graph(img, G, out_png):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    for u,v,data in G.edges(data=True):
        cl = np.array(data["centerline"])
        plt.plot(cl[:,1], cl[:,0], linewidth=1)
    for n,data in G.nodes(data=True):
        plt.plot(data["x"], data["y"], marker='o', markersize=2)
    plt.axis('off'); plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight', pad_inches=0); plt.close()
def build_refined_mask(
    vessel_rgb: np.ndarray,
    px: float,
    py: float,
    collateral_ratio: float,
    collateral_bw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]], Dict[int, Dict[str, object]], Dict[int, List[int]]]:
    vessel_bw = (vessel_rgb[:, :, 0] > 0) | (vessel_rgb[:, :, 1] > 0) | (vessel_rgb[:, :, 2] > 0)
    if not vessel_bw.any():
        refined = np.zeros_like(vessel_rgb, dtype=np.uint8)
        return refined, np.zeros(vessel_bw.shape, dtype=np.int32), [], {}, {}

    skel0 = skeletonize_2d(vessel_bw)
    G = topology_extract(skel0)
    edge_id_img = pixel_edge_assignment(vessel_bw, G, px, py)
    if edge_id_img.max() == 0:
        edge_id_img[vessel_bw] = 1

    stats = compute_node_features(edge_id_img, G, px=px, py=py)
    eid2int = build_eid2int(G)
    adj = edge_adjacency(G, eid2int)

    coll_edges = None
    if collateral_bw is not None:
        edge_ids = edge_id_img.ravel()
        edge_counts = np.bincount(edge_ids, minlength=int(edge_id_img.max()) + 1)
        coll_ids = edge_id_img[collateral_bw].ravel()
        coll_counts = np.bincount(coll_ids, minlength=edge_counts.size)
        coll_edges = []
        for eid in range(1, edge_counts.size):
            if edge_counts[eid] == 0:
                continue
            ratio = float(coll_counts[eid]) / float(edge_counts[eid])
            if ratio > collateral_ratio:
                coll_edges.append(eid)
        if len(coll_edges) == 0:
            coll_edges = []
        coll_set = set(coll_edges)
    else:
        coll_set = set()

    refined = np.zeros_like(vessel_rgb, dtype=np.uint8)
    vessel = edge_id_img > 0
    if coll_set:
        coll_mask = np.isin(edge_id_img, np.array(coll_edges, dtype=np.int32))
        refined[vessel & (~coll_mask)] = (255, 255, 255)
        refined[coll_mask] = (0, 255, 0)
    else:
        refined[vessel] = (255, 255, 255)

    return refined, edge_id_img, coll_edges, stats, adj


def process_one(
    path: Path,
    out_dir: Path,
    json_dir: Path,
    px: float,
    py: float,
    collateral_ratio: float,
    collateral_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    out_json = json_dir / f"{path.stem}_graph.json"
    out_refined = out_dir / path.name
    if (not overwrite) and out_json.exists():
        return out_json
    rgb = load_rgb_mask(path)
    collateral_bw = None
    if collateral_dir is not None:
        cpath = collateral_dir / path.name
        if cpath.exists():
            cimg = load_rgb_mask(cpath)
            # collateral mask is binary (0/255) regardless of channel
            if cimg.ndim == 3:
                collateral_bw = (cimg[:, :, 0] > 0) | (cimg[:, :, 1] > 0) | (cimg[:, :, 2] > 0)
            else:
                collateral_bw = cimg > 0
    refined, edge_id_img, coll_edges, stats, adj = build_refined_mask(rgb, px, py, collateral_ratio, collateral_bw)
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite or (not out_refined.exists()):
        Image.fromarray(refined).save(out_refined)

    edge_ids = sorted(stats.keys())
    if not edge_ids:
        return None

    adj_map: Dict[int, set] = {eid: set(adj.get(eid, [])) for eid in edge_ids}
    centers = {eid: stats[eid]["center"] for eid in edge_ids}

    # connect isolated nodes to nearest center
    for eid in edge_ids:
        if len(adj_map[eid]) == 0 and len(edge_ids) > 1:
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

    # connect all components to make a single connected graph
    adj_map = connect_components(adj_map, centers)

    edge_list = set()
    for u, neighs in adj_map.items():
        for v in neighs:
            if u == v:
                continue
            edge_list.add(tuple(sorted((u, v))))
    edge_list = sorted(list(edge_list))

    nodes = []
    for eid in edge_ids:
        entry = dict(stats[eid])
        entry["id"] = int(eid)
        if coll_edges is not None:
            entry["collateral"] = int(eid in set(coll_edges))
        nodes.append(entry)

    record = {
        "source": path.name,
        "edge_id_img_shape": [int(edge_id_img.shape[0]), int(edge_id_img.shape[1])],
        "nodes": nodes,
        "edges": edge_list,
        "num_nodes": len(nodes),
        "num_edges": len(edge_list),
    }
    if coll_edges is not None:
        record["collateral_edges"] = coll_edges
    json_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(record, indent=2))
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine collateral by vessel partition.")
    ap.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/juncao/data/big_storage/jyc/data/Zurich_DSA_STG/Mask"),
        help="Input mask directory (RGB overlay masks).",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/juncao/data/big_storage/jyc/data/Zurich_DSA_STG/Mask_Graph"),
        help="Output directory for refined masks.",
    )
    ap.add_argument(
        "--json_dir",
        type=Path,
        default=None,
        help="Directory for edge json (default: output_dir).",
    )
    ap.add_argument(
        "--collateral_dir",
        type=Path,
        default=None,
        help="Optional collateral mask directory (binary 0/255). If provided and file exists, add collateral labels.",
    )
    ap.add_argument("--pattern", type=str, default="s*.png", help="Glob pattern for input files.")
    ap.add_argument("--px", type=float, default=1.0, help="pixel size in x (col)")
    ap.add_argument("--py", type=float, default=1.0, help="pixel size in y (row)")
    ap.add_argument(
        "--collateral_ratio",
        type=float,
        default=0.1,
        help="Edge is collateral only if green_ratio > this value.",
    )
    ap.add_argument("--num_workers", type=int, default=16, help="Process workers.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern} in {args.input_dir}")

    json_dir = args.json_dir or args.output_dir

    # write generate_script.txt early for traceability
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    script_path = str(Path(__file__).resolve())
    (args.output_dir / "generate_script.txt").write_text(script_path + "\n")
    if json_dir != args.output_dir:
        (json_dir / "generate_script.txt").write_text(script_path + "\n")

    if args.num_workers <= 1:
        for p in tqdm(files, desc="Build graphs"):
            process_one(p, args.output_dir, json_dir, args.px, args.py, args.collateral_ratio, args.collateral_dir, args.overwrite)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [
                ex.submit(process_one, p, args.output_dir, json_dir, args.px, args.py, args.collateral_ratio, args.collateral_dir, args.overwrite)
                for p in files
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Build graphs"):
                f.result()

    print(f"[DONE] refined masks -> {args.output_dir}")
    print(f"[DONE] graph json -> {json_dir} (for all masks)")


if __name__ == "__main__":
    main()
