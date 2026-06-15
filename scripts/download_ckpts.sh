#!/usr/bin/env bash
# Download pretrained LMC checkpoints into ckpt/.
#
# TODO(maintainers): replace the URL placeholders below with the real
# HuggingFace Hub / Zenodo / GitHub-Release-Asset URLs once the artefact
# upload is finalised. Until that is done, copy ckpt/ from the private
# tree manually.
#
# Expected layout produced by this script:
#   ckpt/
#     dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
#     vessel_seg_nnunet/{plans.json,dataset.json,fold_{0..4}/checkpoint_best.pth}
#     pixel_branch_nnunet/{plans.json,dataset.json,fold_{0..4}/checkpoint_best_prauc.pth}
#     graph_branch_gat/{train_config.json,fold{0..4}_best_prauc.pt}
#
# Total size: ~11 GB. Make sure you have the disk space and network
# bandwidth before running.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/ckpt"
mkdir -p "${CKPT_DIR}"

# === REPLACE THESE WITH THE REAL HOSTED-ARCHIVE URL ============================
# Single tarball is recommended (one resumable download, simple checksum).
ARCHIVE_URL="https://example.invalid/lmc_ckpts_v1.tar.gz"     # TODO
ARCHIVE_SHA256="REPLACE_ME_WITH_REAL_SHA256_HASH"             # TODO
# ==============================================================================

echo "[download_ckpts] Target: ${CKPT_DIR}"
echo "[download_ckpts] Source: ${ARCHIVE_URL}"

if [[ "${ARCHIVE_URL}" == https://example.invalid/* ]]; then
    cat <<'EOF' >&2

ERROR: download_ckpts.sh is a stub.

The artefact-store URL has not been wired up yet. Until the maintainers
upload the checkpoints to a public host (HuggingFace Hub / Zenodo / GitHub
Release Assets) and update this script, copy ckpt/ from your private
working tree manually, e.g.:

    rsync -av --progress /path/to/private/ckpt/ ./ckpt/

Then re-run inference.

EOF
    exit 1
fi

TMP_TAR="$(mktemp -t lmc_ckpts.XXXXXX.tar.gz)"
trap 'rm -f "${TMP_TAR}"' EXIT

curl -L --fail --progress-bar -o "${TMP_TAR}" "${ARCHIVE_URL}"

echo "[download_ckpts] Verifying SHA-256 ..."
echo "${ARCHIVE_SHA256}  ${TMP_TAR}" | sha256sum --check --quiet

echo "[download_ckpts] Extracting to ${CKPT_DIR} ..."
tar -xzf "${TMP_TAR}" -C "${CKPT_DIR}" --strip-components=1

echo "[download_ckpts] Done. Layout:"
find "${CKPT_DIR}" -maxdepth 2 -type d | sort
