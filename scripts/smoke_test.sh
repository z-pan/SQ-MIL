#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — fast local CPU sanity check
#
# Runs a minimal Stage 1 pass: CPU only, 1 epoch, a few slides per split,
# writing throwaway artifacts to results/_smoke. Finishes in seconds and
# exercises the whole path (config -> data loading -> model -> loss ->
# checkpoint), catching shape/path/dtype bugs BEFORE you push or train on A100.
#
# Usage:
#   bash scripts/smoke_test.sh                       # Stage 1, fold 0, Conch config
#   bash scripts/smoke_test.sh --config configs/ovarian_conch_s2.yaml --stage 2
#   bash scripts/smoke_test.sh --slides 5            # more slides per split
#
# Requires local data under data/ (embeddings, superpixels, splits, labels.csv)
# and `pip install -r requirements.txt`. No GPU needed.
# =============================================================================
set -euo pipefail

CONFIG="configs/ovarian_conch_s1.yaml"
STAGE="1"
FOLD=0
SLIDES=3
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --stage)  STAGE="$2";  shift 2 ;;
        --fold)   FOLD="$2";   shift 2 ;;
        --slides) SLIDES="$2"; shift 2 ;;
        *)        EXTRA+=("$1"); shift ;;   # pass anything else through to train.py
    esac
done

# Resolve repo root so this works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== SMOKE: config=$CONFIG stage=$STAGE fold=$FOLD slides=$SLIDES (CPU, 1 epoch) =="
python scripts/train.py \
    --config "$CONFIG" \
    --stage "$STAGE" \
    --fold "$FOLD" \
    --fast_dev_run \
    --limit_slides "$SLIDES" \
    "${EXTRA[@]}"

echo "== SMOKE PASSED =="
