#!/usr/bin/env bash
# =============================================================================
# Script 05 — Train SMMILe Stage 2
#
# Loads the Stage 1 checkpoint specified in the config (paths.stage1_ckpt)
# and fine-tunes with instance refinement + MRF loss.
#
# Usage:
#   bash scripts/05_train_stage2.sh [OPTIONS]
#
# Options:
#   --config        Path to Stage 2 YAML config  (default: configs/ovarian_conch_s2.yaml)
#   --fold          Cross-validation fold index   (default: 0)
#   --stage1_ckpt   Override Stage 1 checkpoint path
#   --gpu           GPU device index              (default: 0)
#   --seed          Random seed                   (default: 42)
# =============================================================================
set -euo pipefail

CONFIG="configs/ovarian_conch_s2.yaml"
FOLD=0
GPU=0
SEED=42
STAGE1_CKPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --fold)         FOLD="$2";         shift 2 ;;
        --gpu)          GPU="$2";          shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --stage1_ckpt)  STAGE1_CKPT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

echo "============================================"
echo "SMMILe Stage 2 Training"
echo "  Config      : ${CONFIG}"
echo "  Fold        : ${FOLD}"
echo "  Stage1 ckpt : ${STAGE1_CKPT:-<from config>}"
echo "  GPU         : ${GPU}"
echo "============================================"

EXTRA_ARGS=""
if [[ -n "${STAGE1_CKPT}" ]]; then
    EXTRA_ARGS="--stage1_ckpt ${STAGE1_CKPT}"
fi

python -m src.training.train_entry \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --stage 2 \
    ${EXTRA_ARGS}

echo "Stage 2 training complete (fold ${FOLD})."
