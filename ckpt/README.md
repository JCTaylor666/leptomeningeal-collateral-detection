# `ckpt/` — pretrained weights

All weights needed by `scripts/infer_one.py`. Total ~11 GB.

## Contents

```
ckpt/
├── dinov3/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth         1.2 GB   ViT-L/16 LVD-1689M backbone
│
├── vessel_seg_nnunet/                                       Sec 4.2: DIAS-trained vessel mask
│   ├── plans.json
│   ├── dataset.json
│   └── fold_{0..4}/checkpoint_best.pth            5×354 MB  clDice loss, 2D nnU-Net
│
├── pixel_branch_nnunet/                                     Sec 4.4: collateral pixel branch
│   ├── plans.json
│   ├── dataset.json
│   └── fold_{0..4}/checkpoint_best_prauc.pth      5×350 MB  PR-AUC selection, masked input
│
└── graph_branch_gat/                                        Sec 4.3: DINOv3 + GAT
    ├── train_config.json
    └── fold{0..4}_best_prauc.pt                   5×1.2 GB  exp_00167 best-PRAUC selection
```

## Provenance

Trained checkpoints are copied verbatim from:

| File group | Source path |
|---|---|
| `dinov3/` | DINOv3 official LVD-1689M ViT-L/16 release |
| `vessel_seg_nnunet/fold_*/checkpoint_best.pth` | Dataset501_DIASVessel, trainer `nnUNetTrainerClDice2D` |
| `pixel_branch_nnunet/fold_*/checkpoint_best_prauc.pth` | Dataset708_ZurichCollateralCVMasked_EVEN, trainer `nnUNetTrainerPRAUC2D` |
| `graph_branch_gat/fold*_best_prauc.pt` | exp_00167_2045694, batch BATCH1 |

## DINOv3 weights filename

The DINOv3 hub loader requires the weights file to be named with an
8-character hash suffix (e.g. `*-8aa4cbdd.pth`; `8aa4cbdd` is the
canonical LVD-1689M hash). We therefore ship the single real weights
file already named `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`, and
the inference config points directly at it — **no symlink or alias is
created** at load time.

## Public-release notes

- Total size (~11 GB) exceeds typical git limits. For public release,
  host these on a dedicated artefact store (HuggingFace Hub / Zenodo /
  GitHub Release Assets) and use `scripts/download_ckpts.sh` (currently
  a stub — fill in the artefact URL and SHA-256) to pull the same
  layout.
