#!/usr/bin/env bash
# =============================================================================
# Script 06 — Evaluate SMMILe on the test split
#
# Runs inference on the test set for a given fold, saves _inst.csv results,
# and prints WSI-level and patch-level metrics.
#
# Usage:
#   bash scripts/06_evaluate.sh [OPTIONS]
#
# Options:
#   --config    Stage 2 YAML config             (default: configs/ovarian_conch_s2.yaml)
#   --fold      Fold index                       (default: 0)
#   --ckpt      Model checkpoint path            (default: results/stage2/fold{fold}/best_model.pth)
#   --gpu       GPU device index                 (default: 0)
#   --all_folds If set, evaluate all 5 folds and aggregate results.
# =============================================================================
set -euo pipefail

CONFIG="configs/ovarian_conch_s2.yaml"
FOLD=0
GPU=0
CKPT=""
ALL_FOLDS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2";    shift 2 ;;
        --fold)      FOLD="$2";      shift 2 ;;
        --gpu)       GPU="$2";       shift 2 ;;
        --ckpt)      CKPT="$2";      shift 2 ;;
        --all_folds) ALL_FOLDS=1;    shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

run_fold() {
    local fold="$1"
    local ckpt="${CKPT:-results/stage2/fold${fold}/best_model.pth}"
    echo "--- Evaluating fold ${fold} | ckpt: ${ckpt} ---"
    python -m src.training.evaluate_entry \
        --config "${CONFIG}" \
        --fold "${fold}" \
        --ckpt "${ckpt}"
}

if [[ "${ALL_FOLDS}" -eq 1 ]]; then
    for fold in 0 1 2 3 4; do
        run_fold "${fold}"
    done
    # Aggregate across folds
    python -c "
import glob, json, numpy as np
files = sorted(glob.glob('results/stage2/fold*/metrics.json'))
if not files:
    print('No metrics.json files found.')
    exit()
keys = None
vals = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    if keys is None:
        keys = list(d.keys())
        vals = {k: [] for k in keys}
    for k in keys:
        vals[k].append(d[k])
print('=== Aggregate results (mean ± std) ===')
for k in keys:
    arr = np.array(vals[k])
    print(f'  {k}: {arr.mean():.4f} ± {arr.std():.4f}')
"
else
    run_fold "${FOLD}"
fi

echo "Evaluation complete."
