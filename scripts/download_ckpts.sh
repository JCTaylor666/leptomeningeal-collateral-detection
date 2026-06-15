#!/usr/bin/env bash
# Download the LMC checkpoints from the HuggingFace Hub into ckpt/.
#
# Source (public, no token needed):
#   https://huggingface.co/cjy666/lmc-ckpt
#     vessel_seg_nnunet/   pixel_branch_nnunet/   graph_branch_gat/
#
# NOT included (gated): the DINOv3 ViT-L/16 backbone. You must request
# access from Meta and place the weights yourself — see the end of this
# script and ckpt/README.md.
#
# Layout produced under ckpt/:
#   vessel_seg_nnunet/{plans.json,dataset.json,fold_{0..4}/checkpoint_best.pth}
#   pixel_branch_nnunet/{plans.json,dataset.json,fold_{0..4}/checkpoint_best_prauc.pth}
#   graph_branch_gat/{train_config.json,fold{0..4}_best_prauc.pt}
#   dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth   <-- YOU add this (gated)
#
# Download size from HF: ~3.5 GB.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/ckpt"
HF_REPO="cjy666/lmc-ckpt"

mkdir -p "${CKPT_DIR}"

echo "[download_ckpts] Downloading ${HF_REPO} -> ${CKPT_DIR}"

# huggingface_hub is the only requirement; install if missing.
python -c "import huggingface_hub" 2>/dev/null || pip install -U "huggingface_hub>=0.23"

# Robust across huggingface_hub versions; public repo => anonymous download.
python - "${HF_REPO}" "${CKPT_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, repo_type="model", local_dir=dst)
print("[download_ckpts] snapshot complete")
PY

cat <<EOF

[download_ckpts] Done — vessel / pixel / graph checkpoints are in ${CKPT_DIR}.

NEXT (required): the DINOv3 backbone is GATED and is NOT distributed here.
  1. Request access to & download the 'dinov3_vitl16_pretrain_lvd1689m'
     (ViT-L/16, LVD-1689M) weights from:
         https://github.com/facebookresearch/dinov3
  2. Place the file at (exact name, with the -8aa4cbdd hash suffix):
         ${CKPT_DIR}/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  3. Clone the DINOv3 source repo and point LMC_DINOV3_REPO at it (see README).
EOF
