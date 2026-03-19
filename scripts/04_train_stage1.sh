#!/usr/bin/env bash
# =============================================================================
# Script 04 — Train SMMILe Stage 1
#
# Usage:
#   bash scripts/04_train_stage1.sh [OPTIONS]
#
# Options:
#   --config   Path to Stage 1 YAML config  (default: configs/ovarian_conch_s1.yaml)
#   --fold     Cross-validation fold index  (default: 0)
#   --gpu      GPU device index             (default: 0)
#   --seed     Random seed                  (default: 42)
# =============================================================================
set -euo pipefail

# ---------- Defaults ----------
CONFIG="configs/ovarian_conch_s1.yaml"
FOLD=0
GPU=0
SEED=42

# ---------- Argument parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --fold)   FOLD="$2";   shift 2 ;;
        --gpu)    GPU="$2";    shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------- Environment ----------
export CUDA_VISIBLE_DEVICES="${GPU}"

# Resolve repo root (script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

echo "============================================"
echo "SMMILe Stage 1 Training"
echo "  Config : ${CONFIG}"
echo "  Fold   : ${FOLD}"
echo "  GPU    : ${GPU}"
echo "  Seed   : ${SEED}"
echo "============================================"

python -m src.training.train_entry \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --stage 1

echo "Stage 1 training complete (fold ${FOLD})."
