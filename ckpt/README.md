# `ckpt/` — pretrained weights

Weights needed by `scripts/infer_one.py`. Everything except the DINOv3
backbone is hosted publicly on the HuggingFace Hub; the DINOv3 backbone is
gated and must be obtained from Meta separately.

## Download

```bash
# vessel / pixel / graph checkpoints (~3.5 GB, public, no token):
bash scripts/download_ckpts.sh
#   -> https://huggingface.co/cjy666/lmc-ckpt  into  ckpt/
```

Then add the **gated** DINOv3 ViT-L/16 backbone yourself (see
[DINOv3 backbone](#dinov3-backbone-gated) below).

## Contents

```
ckpt/
├── dinov3/                                                 NOT on HF — gated, you add it
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
    └── fold{0..4}_best_prauc.pt                   5×3.6 MB  GAT heads only (see note)
```

> **Graph-branch checkpoints are DINOv3-free.** The GAT head is only ~3.6 MB;
> the frozen DINOv3 backbone is loaded separately from `ckpt/dinov3/` at
> runtime (and is intentionally stripped from these checkpoints so the gated
> backbone is never redistributed). Inference is bit-identical either way.

## DINOv3 backbone (gated)

The DINOv3 weights are **not** redistributed here. To get them:

1. Request access to and download `dinov3_vitl16_pretrain_lvd1689m`
   (ViT-L/16, LVD-1689M) from <https://github.com/facebookresearch/dinov3>.
2. Place the file at `ckpt/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`.
   The DINOv3 hub loader requires the 8-character hash suffix (`-8aa4cbdd`,
   the canonical LVD-1689M hash); the inference config points directly at this
   filename — **no symlink or alias is created** at load time.
3. Clone the DINOv3 source repo and export `LMC_DINOV3_REPO` (see the top-level
   README).

## Provenance

| File group | Source |
|---|---|
| `dinov3/` | DINOv3 official LVD-1689M ViT-L/16 release (gated; user-obtained) |
| `vessel_seg_nnunet/fold_*/checkpoint_best.pth` | Dataset501_DIASVessel, trainer `nnUNetTrainerClDice2D` |
| `pixel_branch_nnunet/fold_*/checkpoint_best_prauc.pth` | Dataset708_ZurichCollateralCVMasked_EVEN, trainer `nnUNetTrainerPRAUC2D` |
| `graph_branch_gat/fold*_best_prauc.pt` | exp_00167_2045694, batch BATCH1 (GAT head, DINOv3 stripped) |
