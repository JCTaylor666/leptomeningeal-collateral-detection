# I/O formats for the LMC inference CLI

The single-frame inference script `scripts/infer_one.py` takes one DSA
frame and emits plain JSON / PNG files. This document is the schema
reference for each one.

## Input

### `--image PATH/<caseid>.png` (required)
- 8-bit grayscale PNG (RGB also accepted; converted to luminance via
  `0.299 R + 0.587 G + 0.114 B`).
- Any 2D resolution. Must contain a single 2D DSA frame.
- The `<caseid>` portion of the filename becomes the output file prefix.

## Outputs

### `<caseid>_pred.json`
Per-node predictions in graph-id order:

```jsonc
{
    "caseid": "<caseid>",
    "image_path": "...",
    "lambda": 0.77,
    "mode": "ensemble" | "fold:N",
    "single_fold": null | <int>,
    "threshold": 0.5,
    "num_nodes": <int>,
    "predictions": [
        {
            "node_id":   <int>,
            "bbox":      [y0, y1, x0, x1],
            "num_pixels": <int>,
            "p_gnn":     <float>,           // graph-branch sigmoid output
            "p_nn":      <float>,           // pixel-branch node-pooled probability
            "p_fuse":    <float>,           // λ p_gnn + (1−λ) p_nn
            "hard_pred": 0 | 1              // p_fuse > threshold
        },
        ...
    ]
}
```

### `<caseid>_overlay.png`
Original DSA frame in RGB with red-tinted boxes / pixels for nodes
predicted positive at `--threshold`, and faint cyan boxes for the rest.

### `<caseid>_vessel_mask.png`
Binary uint8 PNG of the predicted vessel mask.

### `<caseid>_graph_pred.json`
The full graph used for inference. Lets downstream tooling reuse the graph
without re-deriving it.

```jsonc
{
    "source": "<caseid>.png",                 // free-form provenance
    "edge_id_img_shape": [H, W],              // image used to lay out the graph
    "num_nodes": <int>,                       // = len(nodes)
    "num_edges": <int>,                       // = len(edges)
    "nodes": [
        {
            "id":          <int>,             // node index, used for sorting + edge refs
            "bbox":        [y0, y1, x0, x1],  // pixel bbox in image space (inclusive y1/x1)
            "center":      [cy, cx],          // region centroid (pixel space)
            "center_skel": [cy, cx],          // centerline centroid
            "center_proj": [cy, cx],          // centerline point closest to centroid
            "pixels":      [[y, x], ...],     // every pixel assigned to this segment
            "bbox_hw":     [bh, bw]
        },
        ...
    ],
    "edges": [
        [u_id, v_id],                         // undirected, refs node "id" fields
        ...
    ]
}
```
