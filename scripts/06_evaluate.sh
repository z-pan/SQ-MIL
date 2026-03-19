#!/usr/bin/env bash
# =============================================================================
# 06_evaluate.sh — SMMILe standalone evaluation
#
# Loads a trained checkpoint and evaluates on the test split.
# Saves per-slide instance prediction CSVs for heatmap generation
# (07_generate_heatmaps.py) and prints WSI + spatial metrics.
#
# Useful when you want to re-evaluate without re-training, or when
# evaluating Stage 1 vs Stage 2 checkpoints side-by-side.
#
# Usage:
#   bash scripts/06_evaluate.sh [OPTIONS]
#
# Options:
#   --config      YAML config (Stage 1 or 2)      (default: configs/ovarian_conch_s2.yaml)
#   --fold        Fold index (0-based)             (default: 0)
#   --all_folds   Evaluate all 5 folds and print aggregate summary
#   --ckpt        Checkpoint path                  (default: results/stage2/fold{N}/best_model.pth)
#   --gpu         GPU device index                 (default: 0)
#   --data_root   Override paths.data_root
# =============================================================================
set -euo pipefail

CONFIG="configs/ovarian_conch_s2.yaml"
FOLD=0
GPU=0
CKPT=""
ALL_FOLDS=0
DATA_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2";    shift 2 ;;
        --fold)      FOLD="$2";      shift 2 ;;
        --gpu)       GPU="$2";       shift 2 ;;
        --ckpt)      CKPT="$2";      shift 2 ;;
        --all_folds) ALL_FOLDS=1;    shift   ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "============================================"
echo " SMMILe — Evaluation"
echo "  Config : ${CONFIG}"
echo "  GPU    : ${GPU}"
[[ "${ALL_FOLDS}" -eq 1 ]] && echo "  Folds  : all (0-4)" || echo "  Fold   : ${FOLD}"
echo "============================================"

EXTRA_ARGS=()
[[ -n "${DATA_ROOT}" ]] && EXTRA_ARGS+=(--data_root "${DATA_ROOT}")

run_fold() {
    local fold="$1"
    local ckpt="${CKPT:-results/stage2/fold${fold}/best_model.pth}"

    if [[ ! -f "${ckpt}" ]]; then
        echo "WARNING: Checkpoint not found: ${ckpt} — skipping fold ${fold}." >&2
        return 0
    fi

    echo ""
    echo "--- Evaluating fold ${fold} | ckpt: ${ckpt} ---"
    python scripts/train.py \
        --config "${CONFIG}" \
        --stage  eval \
        --fold   "${fold}" \
        --ckpt   "${ckpt}" \
        "${EXTRA_ARGS[@]}"
    echo "Fold ${fold} evaluation complete."
}

if [[ "${ALL_FOLDS}" -eq 1 ]]; then
    for fold in 0 1 2 3 4; do
        run_fold "${fold}"
    done
    echo ""
    # Aggregate metrics from saved JSON files
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
    echo ""
    echo "All folds evaluated."
else
    run_fold "${FOLD}"
fi

echo "Evaluation complete."
