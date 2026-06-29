# LMC — Leptomeningeal Collateral Detection on DSA

Inference code for *Leptomeningeal Collateral Detection on DSA via
Vessel-Graph Neural Networks* (MIUA 2026).

<p align="center">
  <img src="demo/demo.gif" alt="LMC live inference demo — drag a DSA frame and watch each pipeline stage render" width="850">
</p>

<p align="center"><em>Interactive web demo — each stage renders live as it is computed. See <a href="demo/README.md">demo/</a>.</em></p>

Feed a single 2D DSA frame; the pipeline predicts a vessel mask, builds a
vessel-segment graph, and outputs a collateral probability for every
vessel segment (graph node).

## Pipeline

```
raw DSA frame (PNG, 2D grayscale)
  │
  ▼  vessel segmentation   ── DIAS-trained 5-fold nnU-Net + clDice
binary vessel mask
  │
  ▼  graph construction    ── line-graph: nodes = vessel segments
graph (nodes: bbox / center_proj / pixels / adjacency)
  ├─► graph branch   ── DINOv3 ViT-L/16 (frozen) + GAT          → p_gnn
  └─► pixel branch   ── vessel-masked 5-fold nnU-Net + node-mean → p_nn
  │
  ▼  fusion   p_fuse = λ·p_gnn + (1−λ)·p_nn   (λ = 0.77)
per-node collateral probability
```

## Setup

```bash
# 1. Environment (conda; tested with CUDA 12.1 / torch 2.3).
conda env create -f environment.yml
conda activate lmc
pip install -r requirements.txt

# 2. Local clone of the DINOv3 source repo (for the ViT-L/16 backbone):
git clone https://github.com/facebookresearch/dinov3 /path/to/dinov3
export LMC_DINOV3_REPO=/path/to/dinov3        # or pass --dino_repo_dir

# 3. Download the checkpoints (vessel / pixel / graph) from the HF Hub:
bash scripts/download_ckpts.sh                # -> ckpt/  (~3.5 GB, public)
#    Then add the GATED DINOv3 backbone yourself (see ckpt/README.md):
#    place dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth under ckpt/dinov3/
```

## Run

```bash
python scripts/infer_one.py \
    --image examples/sample_frame/aaaaaa.png \
    --output_dir out/aaaaaa \
    --threshold 0.5
```

Outputs (under `--output_dir`):

- `<caseid>_pred.json` — per-node probabilities (`p_gnn`, `p_nn`,
  `p_fuse`, `hard_pred`) plus bounding box and pixel count.
- `<caseid>_overlay.png` — DSA frame with predicted collateral nodes highlighted.
- `<caseid>_vessel_mask.png` — predicted binary vessel mask.
- `<caseid>_graph_pred.json` — full predicted graph topology.

Single-fold inference (skip the 5-fold ensemble):

```bash
python scripts/infer_one.py --image ... --output_dir ... --mode fold:0
```

For the full I/O schemas see [`docs/io_formats.md`](docs/io_formats.md).

## Interactive demo (web UI)

The repo also ships a browser-based visualization of the full pipeline.
Drag a DSA frame into the page, press **Start**, and every stage renders as
soon as it is computed — vessel mask → graph → masked input → pixel / graph
branches → fusion → collateral decision — streamed live to an animated
pipeline diagram.

```bash
pip install -r demo/requirements.txt          # adds flask
export LMC_DINOV3_REPO=/path/to/dinov3         # same as CLI inference
python demo/app.py                             # → http://127.0.0.1:5000
```

It reuses the exact same runners as `scripts/infer_one.py` — no retraining,
no extra weights. See **[`demo/README.md`](demo/README.md)** for the full
deployment guide (port options, remote / SSH / VSCode forwarding, GPU notes,
and troubleshooting).

## Model

- Backbone: DINOv3 ViT-L/16 (LVD-1689M), frozen.
- Graph branch: GAT, 2 layers, hidden dim 128; ROI-Align features + bbox H/W.
- Pixel branch: 2D nnU-Net on vessel-masked frames, pooled to nodes.
- Fusion weight: λ = 0.77.

## Checkpoints

Hosted publicly at **[`cjy666/lmc-ckpt`](https://huggingface.co/cjy666/lmc-ckpt)**
(~3.5 GB) and fetched by `scripts/download_ckpts.sh`:

| Subdir | Contents | Source |
|---|---|---|
| `ckpt/vessel_seg_nnunet/`   | 5 folds × `checkpoint_best.pth` (DIAS, clDice) | HF Hub |
| `ckpt/pixel_branch_nnunet/` | 5 folds × `checkpoint_best_prauc.pth` (vessel-masked) | HF Hub |
| `ckpt/graph_branch_gat/`    | 5 folds × `fold{i}_best_prauc.pt` (GAT head, ~3.6 MB) | HF Hub |
| `ckpt/dinov3/`              | DINOv3 ViT-L/16 LVD-1689M backbone | **gated — obtain from [Meta](https://github.com/facebookresearch/dinov3)** |

The DINOv3 backbone is **not** redistributed (gated license); the graph-branch
checkpoints ship the GAT head only, with the frozen backbone stripped — it is
loaded from `ckpt/dinov3/` at runtime (inference is bit-identical). See
[`ckpt/README.md`](ckpt/README.md) for the full layout and the DINOv3 steps.

## Layout

```
public_repo/
├── scripts/
│   ├── infer_one.py              single-frame inference CLI
│   └── download_ckpts.sh         checkpoint download helper
├── src/lmc/
│   ├── inference/                end-to-end runner (vessel → graph → branches → fusion)
│   ├── preprocess/mask_to_graph.py   line-graph construction
│   └── graph_branch/{backbone,model}.py   DINOv3 + GAT
├── configs/inference_default.yaml
├── ckpt/                         pretrained weights (see ckpt/README.md)
├── demo/                         interactive web UI (Flask + HTML); see demo/README.md
├── examples/
│   └── sample_frame/aaaaaa.png
├── docs/io_formats.md
└── tests/test_smoke.py
```

## License / citation

See `LICENSE`. Cite as:

```bibtex
@misc{cao2026leptomeningealcollateraldetectiondsa,
      title={Leptomeningeal Collateral Detection on DSA via Vessel-Graph Neural Networks},
      author={Junyong Cao and Hakim Baazaoui and Chinmay Prabhakar and Suprosanna Shit and Lukas Bastian Otto and Susanne Wegener and Bjoern Menze and Ezequiel de la Rosa},
      year={2026},
      eprint={2606.14828},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2606.14828},
}
```
