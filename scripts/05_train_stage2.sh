#!/usr/bin/env bash
# =============================================================================
# 05_train_stage2.sh — SMMILe Stage 2 training
#
# Loads Stage 1 weights and jointly trains the full model with
# L_cls + L_ref + L_mrf.  Evaluation on the test split is run automatically
# at the end of each fold.
#
# Usage:
#   bash scripts/05_train_stage2.sh [OPTIONS]
#
# Options:
#   --config        Stage 2 YAML config             (default: configs/ovarian_conch_s2.yaml)
#   --fold          Single fold index (0-based)      (default: 0)
#   --all_folds     Run all 5 folds sequentially
#   --stage1_ckpt   Stage 1 checkpoint path          (default: results/stage1/fold{N}/best_model.pth)
#   --gpu           GPU device index                 (default: 0)
#   --seed          Random seed override
#   --data_root     Override paths.data_root
#   --output_dir    Override paths.output_dir
#   --epochs        Override training.epochs
# =============================================================================
set -euo pipefail

CONFIG="configs/ovarian_conch_s2.yaml"
FOLD=0
GPU=0
SEED=""
STAGE1_CKPT=""
DATA_ROOT=""
OUTPUT_DIR=""
EPOCHS=""
ALL_FOLDS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";       shift 2 ;;
        --fold)         FOLD="$2";         shift 2 ;;
        --gpu)          GPU="$2";          shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --stage1_ckpt)  STAGE1_CKPT="$2";  shift 2 ;;
        --data_root)    DATA_ROOT="$2";    shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --epochs)       EPOCHS="$2";       shift 2 ;;
        --all_folds)    ALL_FOLDS=1;       shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "============================================"
echo " SMMILe — Stage 2 Training"
echo "  Config     : ${CONFIG}"
echo "  GPU        : ${GPU}"
[[ "${ALL_FOLDS}" -eq 1 ]] && echo "  Folds      : all (0-4)" || echo "  Fold       : ${FOLD}"
echo "  Stage1 ckpt: ${STAGE1_CKPT:-<auto: results/stage1/fold{N}/best_model.pth>}"
echo "============================================"

# ---------- Build common args ----------
EXTRA_ARGS=()
[[ -n "${SEED}"        ]] && EXTRA_ARGS+=(--seed        "${SEED}")
[[ -n "${DATA_ROOT}"   ]] && EXTRA_ARGS+=(--data_root   "${DATA_ROOT}")
[[ -n "${OUTPUT_DIR}"  ]] && EXTRA_ARGS+=(--output_dir  "${OUTPUT_DIR}")
[[ -n "${EPOCHS}"      ]] && EXTRA_ARGS+=(--epochs      "${EPOCHS}")

run_fold() {
    local fold="$1"
    # Stage 1 checkpoint: use override or derive per-fold path automatically
    local s1_ckpt="${STAGE1_CKPT:-results/stage1/fold${fold}/best_model.pth}"

    if [[ ! -f "${s1_ckpt}" ]]; then
        echo "ERROR: Stage 1 checkpoint not found: ${s1_ckpt}" >&2
        echo "       Run 04_train_stage1.sh for fold ${fold} first." >&2
        exit 1
    fi

    echo ""
    echo "--- Stage 2 | Fold ${fold} | Stage1 ckpt: ${s1_ckpt} ---"
    python scripts/train.py \
        --config      "${CONFIG}" \
        --stage       2 \
        --fold        "${fold}" \
        --stage1_ckpt "${s1_ckpt}" \
        "${EXTRA_ARGS[@]}"
    echo "Stage 2 fold ${fold} complete."
}

if [[ "${ALL_FOLDS}" -eq 1 ]]; then
    for fold in 0 1 2 3 4; do
        run_fold "${fold}"
    done
    echo ""
    echo "All Stage 2 folds complete."
    # Print aggregated summary (train.py already logged per-fold metrics JSON)
    python -c "
import json, glob, numpy as np, sys
files = sorted(glob.glob('results/stage2/fold*/eval_metrics.json'))
if not files:
    print('No eval_metrics.json files found under results/stage2/.')
    sys.exit(0)
keys = ['wsi_auc','patch_auc','patch_f1','patch_acc','patch_precision','patch_recall']
vals = {k: [] for k in keys}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    for k in keys:
        vals[k].append(d.get(k, float('nan')))
print('===== 5-fold aggregate results (mean ± std) =====')
for k in keys:
    arr = np.array(vals[k])
    print(f'  {k:20s}: {arr.mean():.4f} ± {arr.std():.4f}')
"
else
    run_fold "${FOLD}"
fi
