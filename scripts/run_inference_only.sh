#!/usr/bin/env bash
# =============================================================================
# run_inference_only.sh — SMMILe inference + heatmap generation for new WSIs.
#
# Applies a previously trained Stage 2 model checkpoint to one or more new
# .tif whole-slide images and produces per-case heatmaps.  No ground-truth
# labels are required — a temporary dummy labels.csv and splits CSV are
# created automatically and cleaned up afterward.
#
# PIPELINE (inference-only mode)
#   1. [optional] Extract patch embeddings for the new WSIs
#   2. [optional] Generate SLIC superpixels
#   3. Create temporary labels.csv and splits CSV (all slides → test split)
#   4. Run evaluation / inference using the trained checkpoint
#       → writes per-slide instance prediction CSV(s)
#   5. Generate subtype-colored heatmap PNGs
#   6. Clean up temporary files
#
# Steps 1–2 are skipped automatically if embeddings / superpixels already
# exist in the target directories.  Use --skip-features / --skip-superpixels
# to force-skip regardless.
#
# USAGE
#   bash scripts/run_inference_only.sh [OPTIONS]
#
# REQUIRED
#   --model_dir PATH  Directory containing best_model.pth for one or more
#                     Stage 2 folds (e.g. results/stage2/).
#                     If the directory contains fold{N}/ sub-dirs the first
#                     fold with a checkpoint is used, unless --fold is set.
#   --wsi_dir PATH    Directory containing the new *.tif WSI files, OR a
#                     single .tif file path.
#
# OPTIONS
#   --fold       N    Fold index of the model to use  [default: 0]
#   --config   PATH   Stage 2 YAML config             [default: configs/ovarian_conch_s2.yaml]
#   --output_dir PATH Output directory for heatmaps   [default: results/heatmaps_infer]
#   --embed_dir PATH  Embeddings directory            [default: {work_dir}/embeddings]
#   --sp_dir    PATH  Superpixels directory           [default: {work_dir}/superpixels]
#   --work_dir  PATH  Scratch directory for temp files [default: /tmp/sq_mil_infer_$$]
#   --encoder   NAME  conch | resnet50                [default: conch]
#   --gpu       N     GPU device index                [default: 0]
#   --patch_size N    Patch edge length (px)          [default: 512]
#   --thumbnail  N    Heatmap thumbnail max dim (px)  [default: 2048]
#   --workers   N     Parallel heatmap workers        [default: 8]
#   --skip-features   Skip embedding extraction (assume already done)
#   --skip-superpixels Skip superpixel generation
#   --keep-tmp        Do NOT delete the scratch directory on exit
#   --dry-run         Print commands without executing
#   -h / --help       Show this help and exit
#
# EXAMPLE — apply fold-0 checkpoint to two new WSIs:
#   bash scripts/run_inference_only.sh \
#       --model_dir results/stage2 \
#       --wsi_dir   /data/new_cases \
#       --output_dir results/new_case_heatmaps \
#       --fold 0 \
#       --gpu  0
#
# EXAMPLE — single .tif file:
#   bash scripts/run_inference_only.sh \
#       --model_dir results/stage2 \
#       --wsi_dir   data/wsi/new_slide.tif \
#       --output_dir results/heatmaps_infer
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RED='\033[0;31m'; _GREEN='\033[0;32m'; _YELLOW='\033[1;33m'
_CYAN='\033[0;36m'; _BOLD='\033[1m'; _NC='\033[0m'

log_info()  { echo -e "${_CYAN}[INFO]${_NC}  $*"; }
log_ok()    { echo -e "${_GREEN}[OK]${_NC}    $*"; }
log_warn()  { echo -e "${_YELLOW}[WARN]${_NC}  $*" >&2; }
log_error() { echo -e "${_RED}[ERROR]${_NC} $*" >&2; }
log_step()  { echo -e "\n${_BOLD}${_CYAN}────────────────────────────────────────${_NC}"; \
              echo -e "${_BOLD}${_CYAN}  $*${_NC}"; \
              echo -e "${_BOLD}${_CYAN}────────────────────────────────────────${_NC}"; }
log_skip()  { echo -e "${_YELLOW}[SKIP]${_NC}  $*"; }

die() { log_error "$*"; exit 1; }

require_file() {
    local path="$1"; local hint="${2:-}"
    [[ -f "$path" ]] && return 0
    log_error "Required file not found: $path"
    [[ -n "$hint" ]] && log_error "  → $hint"
    exit 1
}

require_dir() {
    local path="$1"; local hint="${2:-}"
    [[ -d "$path" ]] && return 0
    log_error "Required directory not found: $path"
    [[ -n "$hint" ]] && log_error "  → $hint"
    exit 1
}

run_cmd() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo -e "${_YELLOW}[DRY-RUN]${_NC}  $*"
    else
        "$@"
    fi
}

usage() {
    sed -n '/^# USAGE/,/^# =====/{ /^# =====/d; s/^# \{0,1\}//; p }' "$0"
    exit 0
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_DIR=""
WSI_DIR=""
FOLD=0
CONFIG="configs/ovarian_conch_s2.yaml"
OUTPUT_DIR="results/heatmaps_infer"
EMBED_DIR=""
SP_DIR=""
WORK_DIR="/tmp/sq_mil_infer_$$"
ENCODER="conch"
GPU=0
PATCH_SIZE=512
THUMBNAIL=2048
WORKERS=8

SKIP_FEATURES=0
SKIP_SUPERPIXELS=0
KEEP_TMP=0
DRY_RUN=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_dir)        MODEL_DIR="$2";     shift 2 ;;
        --wsi_dir)          WSI_DIR="$2";       shift 2 ;;
        --fold)             FOLD="$2";          shift 2 ;;
        --config)           CONFIG="$2";        shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2";    shift 2 ;;
        --embed_dir)        EMBED_DIR="$2";     shift 2 ;;
        --sp_dir)           SP_DIR="$2";        shift 2 ;;
        --work_dir)         WORK_DIR="$2";      shift 2 ;;
        --encoder)          ENCODER="$2";       shift 2 ;;
        --gpu)              GPU="$2";           shift 2 ;;
        --patch_size)       PATCH_SIZE="$2";    shift 2 ;;
        --thumbnail)        THUMBNAIL="$2";     shift 2 ;;
        --workers)          WORKERS="$2";       shift 2 ;;
        --skip-features)    SKIP_FEATURES=1;    shift   ;;
        --skip-superpixels) SKIP_SUPERPIXELS=1; shift   ;;
        --keep-tmp)         KEEP_TMP=1;         shift   ;;
        --dry-run)          DRY_RUN=1;          shift   ;;
        -h|--help)          usage ;;
        *) die "Unknown option: $1  (run with --help for usage)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate required arguments
# ---------------------------------------------------------------------------
[[ -z "${MODEL_DIR}" ]] && die "--model_dir is required.
  Specify the directory containing Stage 2 best_model.pth checkpoints.
  Example: --model_dir results/stage2"

[[ -z "${WSI_DIR}" ]] && die "--wsi_dir is required.
  Specify a directory of .tif files or a single .tif file path.
  Example: --wsi_dir data/new_cases"

# ---------------------------------------------------------------------------
# Resolve repo root + paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"

# Embed / superpixel dirs: default to work_dir subdirs
[[ -z "${EMBED_DIR}" ]] && EMBED_DIR="${WORK_DIR}/embeddings"
[[ -z "${SP_DIR}"    ]] && SP_DIR="${WORK_DIR}/superpixels"

TMP_LABELS="${WORK_DIR}/labels.csv"
TMP_SPLITS_DIR="${WORK_DIR}/splits"
TMP_OUTPUT="${WORK_DIR}/predictions"

# ---------------------------------------------------------------------------
# Resolve WSI paths
# ---------------------------------------------------------------------------
if [[ -f "${WSI_DIR}" ]]; then
    # Single .tif file
    WSI_FILE="${WSI_DIR}"
    WSI_ACTUAL_DIR="$(dirname "${WSI_FILE}")"
    WSI_GLOB="${WSI_FILE}"
    SLIDE_IDS=("$(basename "${WSI_FILE}" .tif)")
elif [[ -d "${WSI_DIR}" ]]; then
    WSI_ACTUAL_DIR="${WSI_DIR}"
    WSI_GLOB="${WSI_DIR}/*.tif"
    mapfile -t _PATHS < <(find "${WSI_DIR}" -maxdepth 1 -name "*.tif" | sort)
    SLIDE_IDS=()
    for p in "${_PATHS[@]}"; do
        SLIDE_IDS+=("$(basename "$p" .tif)")
    done
else
    die "wsi_dir does not exist or is not a .tif file: ${WSI_DIR}"
fi

[[ ${#SLIDE_IDS[@]} -eq 0 ]] && die "No .tif files found in: ${WSI_DIR}"

# ---------------------------------------------------------------------------
# Resolve checkpoint path
# ---------------------------------------------------------------------------
_resolve_checkpoint() {
    local base_dir="$1"; local fold="$2"
    # Try fold subdirectory first, then base directory directly
    local candidates=(
        "${base_dir}/fold${fold}/best_model.pth"
        "${base_dir}/best_model.pth"
    )
    for c in "${candidates[@]}"; do
        [[ -f "$c" ]] && echo "$c" && return 0
    done
    return 1
}

CKPT="$(_resolve_checkpoint "${MODEL_DIR}" "${FOLD}" || true)"
if [[ -z "${CKPT}" ]]; then
    die "No checkpoint found for fold ${FOLD} in: ${MODEL_DIR}
  Expected one of:
    ${MODEL_DIR}/fold${FOLD}/best_model.pth
    ${MODEL_DIR}/best_model.pth
  → Train a Stage 2 model first with scripts/05_train_stage2.sh."
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo -e "${_BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║       SMMILe — Inference + Heatmap Pipeline         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${_NC}"
echo "  WSI dir      : ${WSI_DIR}"
echo "  Slides       : ${#SLIDE_IDS[@]}"
echo "  Checkpoint   : ${CKPT}"
echo "  Config       : ${CONFIG}"
echo "  Encoder      : ${ENCODER}"
echo "  GPU          : ${GPU}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "  Scratch dir  : ${WORK_DIR}"
[[ "${DRY_RUN}" -eq 1 ]] && echo "  *** DRY RUN — no commands will be executed ***"
echo ""

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------
_cleanup() {
    if [[ "${KEEP_TMP}" -eq 0 && -d "${WORK_DIR}" && "${DRY_RUN}" -eq 0 ]]; then
        log_info "Removing scratch directory: ${WORK_DIR}"
        rm -rf "${WORK_DIR}"
    fi
}
trap _cleanup EXIT

# ---------------------------------------------------------------------------
# STEP 1 — Feature extraction
# ---------------------------------------------------------------------------
log_step "STEP 1 / 5 — Feature extraction"

if [[ "${SKIP_FEATURES}" -eq 1 ]]; then
    log_skip "Step 1 skipped (--skip-features)."
    # User is responsible for providing --embed_dir pointing to existing data
    require_dir "${EMBED_DIR}" \
        "When using --skip-features, set --embed_dir to the existing embeddings directory."
else
    n_existing=$(find "${EMBED_DIR}" -name "coords.csv" 2>/dev/null | wc -l)
    if [[ "${n_existing}" -ge "${#SLIDE_IDS[@]}" ]]; then
        log_skip "All ${#SLIDE_IDS[@]} slide embedding(s) already exist in ${EMBED_DIR} — skipping."
    else
        mkdir -p "${EMBED_DIR}"
        run_cmd python scripts/01_extract_features.py \
            --encoder_name  "${ENCODER}" \
            --wsi_dir       "${WSI_ACTUAL_DIR}" \
            --output_dir    "${EMBED_DIR}" \
            --patch_size    "${PATCH_SIZE}" \
            --step_size     "${PATCH_SIZE}" \
            --device        "cuda:${GPU}"
        log_ok "Embeddings saved → ${EMBED_DIR}"
    fi
fi

# ---------------------------------------------------------------------------
# STEP 2 — Superpixel generation
# ---------------------------------------------------------------------------
log_step "STEP 2 / 5 — Superpixel generation"

if [[ "${SKIP_SUPERPIXELS}" -eq 1 ]]; then
    log_skip "Step 2 skipped (--skip-superpixels)."
    require_dir "${SP_DIR}" \
        "When using --skip-superpixels, set --sp_dir to the existing superpixels directory."
else
    n_existing_sp=$(find "${SP_DIR}" -name "*.npy" 2>/dev/null | wc -l)
    if [[ "${n_existing_sp}" -ge "${#SLIDE_IDS[@]}" ]]; then
        log_skip "All ${#SLIDE_IDS[@]} superpixel file(s) already exist in ${SP_DIR} — skipping."
    else
        require_dir "${EMBED_DIR}" \
            "Embeddings not found.  Run step 1 first or use --skip-features with --embed_dir."
        mkdir -p "${SP_DIR}"
        run_cmd python scripts/02_generate_superpixels.py \
            --embedding_dir "${EMBED_DIR}" \
            --output_dir    "${SP_DIR}" \
            --patch_size    "${PATCH_SIZE}"
        log_ok "Superpixels saved → ${SP_DIR}"
    fi
fi

# ---------------------------------------------------------------------------
# STEP 3 — Create temporary labels + splits (all slides → test)
# ---------------------------------------------------------------------------
log_step "STEP 3 / 5 — Prepare temporary inference splits"

mkdir -p "${WORK_DIR}" "${TMP_SPLITS_DIR}"

# Write dummy labels.csv (all slides labeled HGSC; label is unused for
# inference but required by the DataLoader).
if [[ "${DRY_RUN}" -eq 0 ]]; then
    {
        echo "slide_id,label"
        for sid in "${SLIDE_IDS[@]}"; do
            echo "${sid},HGSC"
        done
    } > "${TMP_LABELS}"
    log_info "Temporary labels.csv written (${#SLIDE_IDS[@]} slides, dummy label=HGSC)."
else
    echo -e "${_YELLOW}[DRY-RUN]${_NC}  Would write ${TMP_LABELS} with ${#SLIDE_IDS[@]} rows."
fi

# Write splits_0.csv: all slides in the test column, empty train/val.
if [[ "${DRY_RUN}" -eq 0 ]]; then
    python3 - <<PYEOF
import csv, os
slides = ${SLIDE_IDS[@]+"${SLIDE_IDS[@]}"}
slide_list = [s.strip() for s in """${SLIDE_IDS[*]}""".split()]
out = os.path.join("${TMP_SPLITS_DIR}", "splits_0.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train", "val", "test"])
    # train / val cols are empty (NaN-padded by writing empty strings)
    for i, sid in enumerate(slide_list):
        w.writerow(["", "", sid])
print(f"  Wrote {out} ({len(slide_list)} test slides)")
PYEOF
else
    echo -e "${_YELLOW}[DRY-RUN]${_NC}  Would write ${TMP_SPLITS_DIR}/splits_0.csv."
fi

# ---------------------------------------------------------------------------
# STEP 4 — Inference / evaluation
# ---------------------------------------------------------------------------
log_step "STEP 4 / 5 — Running inference with Stage 2 checkpoint"

log_info "Checkpoint : ${CKPT}"
log_info "Config     : ${CONFIG}"
log_info "Output     : ${TMP_OUTPUT}"

mkdir -p "${TMP_OUTPUT}"

run_cmd python scripts/train.py \
    --config       "${CONFIG}" \
    --stage        eval \
    --fold         0 \
    --ckpt         "${CKPT}" \
    --data_root    "${WORK_DIR}" \
    --output_dir   "${TMP_OUTPUT}"

# Check that the prediction CSV was created
PRED_CSV="${TMP_OUTPUT}/fold0/inst_predictions_fold0.csv"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ ! -f "${PRED_CSV}" ]]; then
        die "Expected instance prediction CSV not found: ${PRED_CSV}
  The evaluation step may have failed.  Check the log output above."
    fi
    n_rows=$(tail -n +2 "${PRED_CSV}" | wc -l)
    log_ok "Instance predictions written → ${PRED_CSV} (${n_rows} patch rows)"
fi

# ---------------------------------------------------------------------------
# STEP 5 — Heatmap generation
# ---------------------------------------------------------------------------
log_step "STEP 5 / 5 — Generating heatmaps"

mkdir -p "${OUTPUT_DIR}"

run_cmd python scripts/07_generate_heatmaps.py \
    --wsi_dir         "${WSI_GLOB}" \
    --predictions_dir "${TMP_OUTPUT}" \
    --output_dir      "${OUTPUT_DIR}" \
    --thumbnail_size  "${THUMBNAIL}" \
    --num_workers     "${WORKERS}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${_BOLD}${_GREEN}══════════════════════════════════════════════════${_NC}"
echo -e "${_BOLD}${_GREEN}  Inference + heatmap generation complete!${_NC}"
echo -e "${_BOLD}${_GREEN}══════════════════════════════════════════════════${_NC}"
echo ""
echo "  Checkpoint used      : ${CKPT}"
echo "  Instance predictions : ${TMP_OUTPUT}/fold0/"
echo "  Heatmap PNGs         : ${OUTPUT_DIR}/"
if [[ "${KEEP_TMP}" -eq 0 ]]; then
    echo "  Scratch dir          : (will be removed on exit)"
else
    echo "  Scratch dir          : ${WORK_DIR} (kept — use --keep-tmp to suppress)"
fi
echo ""

if [[ "${DRY_RUN}" -eq 0 ]]; then
    n_out=$(find "${OUTPUT_DIR}" -name "*_heatmap.png" | wc -l)
    log_ok "${n_out} heatmap(s) in ${OUTPUT_DIR}/"
fi
