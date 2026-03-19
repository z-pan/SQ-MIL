#!/usr/bin/env bash
# =============================================================================
# 04_train_stage1.sh — SMMILe Stage 1 training
#
# Trains the primary network (NIC + GatedAttention + InD/InS) with L_cls.
# Run once per fold, or use --all_folds to train all five sequentially.
#
# Usage:
#   bash scripts/04_train_stage1.sh [OPTIONS]
#
# Options:
#   --config      Path to Stage 1 YAML config  (default: configs/ovarian_conch_s1.yaml)
#   --fold        Single fold index (0-based)   (default: 0)
#   --all_folds   Run all 5 folds sequentially
#   --gpu         GPU device index              (default: 0)
#   --seed        Random seed override
#   --data_root   Override paths.data_root
#   --output_dir  Override paths.output_dir
#   --epochs      Override training.epochs (e.g. 200 for ResNet-50)
# =============================================================================
set -euo pipefail

# ---------- Defaults ----------
CONFIG="configs/ovarian_conch_s1.yaml"
FOLD=0
GPU=0
SEED=""
DATA_ROOT=""
OUTPUT_DIR=""
EPOCHS=""
ALL_FOLDS=0

# ---------- Argument parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2";     shift 2 ;;
        --fold)       FOLD="$2";       shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --data_root)  DATA_ROOT="$2";  shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --all_folds)  ALL_FOLDS=1;     shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------- Resolve repo root ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "============================================"
echo " SMMILe — Stage 1 Training"
echo "  Config     : ${CONFIG}"
echo "  GPU        : ${GPU}"
[[ "${ALL_FOLDS}" -eq 1 ]] && echo "  Folds      : all (0-4)" || echo "  Fold       : ${FOLD}"
echo "============================================"

# ---------- Build common args ----------
EXTRA_ARGS=()
[[ -n "${SEED}"       ]] && EXTRA_ARGS+=(--seed       "${SEED}")
[[ -n "${DATA_ROOT}"  ]] && EXTRA_ARGS+=(--data_root  "${DATA_ROOT}")
[[ -n "${OUTPUT_DIR}" ]] && EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${EPOCHS}"     ]] && EXTRA_ARGS+=(--epochs     "${EPOCHS}")

run_fold() {
    local fold="$1"
    echo ""
    echo "--- Stage 1 | Fold ${fold} ---"
    python scripts/train.py \
        --config "${CONFIG}" \
        --stage  1 \
        --fold   "${fold}" \
        "${EXTRA_ARGS[@]}"
    echo "Stage 1 fold ${fold} complete."
}

if [[ "${ALL_FOLDS}" -eq 1 ]]; then
    for fold in 0 1 2 3 4; do
        run_fold "${fold}"
    done
    echo ""
    echo "All Stage 1 folds complete."
else
    run_fold "${FOLD}"
fi
