#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end SMMILe pipeline for ovarian cancer subtype
# classification on .tif whole-slide images.
#
# STEPS
#   1. Feature extraction   (scripts/01_extract_features.py)
#   2. Superpixel generation (scripts/02_generate_superpixels.py)
#   3. Split preparation    (scripts/03_prepare_splits.py)
#   4. Stage 1 training     (scripts/train.py  --stage 1, all folds)
#   5. Stage 2 training     (scripts/train.py  --stage 2, all folds)
#   6. Evaluation           (run automatically at end of Stage 2 per fold)
#   7. Heatmap generation   (scripts/07_generate_heatmaps.py)
#
# Any step can be skipped with its corresponding --skip-* flag, or with
# --resume which automatically skips any step whose output already exists.
#
# USAGE
#   bash scripts/run_pipeline.sh [OPTIONS]
#
# REQUIRED FILES (before running)
#   data/wsi/           — .tif WSI files
#   data/labels.csv     — columns: slide_id, label  (CC/EC/HGSC/LGSC/MC)
#
# QUICK START (all defaults, Conch encoder, GPU 0)
#   bash scripts/run_pipeline.sh --data_root data --gpu 0
#
# TYPICAL PRODUCTION RUN (ResNet-50, skip feature extraction if done)
#   bash scripts/run_pipeline.sh \
#       --encoder resnet50 \
#       --skip-features \
#       --gpu 0
#
# OPTIONS
#   --data_root PATH    Root data directory           [default: data]
#   --wsi_dir   PATH    Directory containing *.tif    [default: {data_root}/wsi]
#   --results   PATH    Root results directory        [default: results]
#   --encoder   NAME    conch | resnet50              [default: conch]
#   --gpu       N       GPU device index              [default: 0]
#   --n_folds   N       Number of CV folds            [default: 5]
#   --seed      N       Global random seed            [default: 42]
#   --s1_config PATH    Stage 1 YAML config           [default: configs/ovarian_conch_s1.yaml]
#   --s2_config PATH    Stage 2 YAML config           [default: configs/ovarian_conch_s2.yaml]
#   --patch_size N      Patch edge length (px)        [default: 512]
#   --thumbnail N       Heatmap thumbnail max dim     [default: 2048]
#   --workers   N       Parallel workers (heatmaps)   [default: 8]
#   --resume            Skip steps whose outputs exist (idempotent re-run)
#   --skip-features     Skip step 1 (feature extraction)
#   --skip-superpixels  Skip step 2 (superpixel generation)
#   --skip-splits       Skip step 3 (split preparation)
#   --skip-training     Skip steps 4+5 (all training)
#   --skip-stage1       Skip step 4 only
#   --skip-stage2       Skip step 5+6 only
#   --skip-heatmaps     Skip step 7 (heatmap generation)
#   --dry-run           Print commands without executing them
#   -h / --help         Show this help and exit
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RED='\033[0;31m'; _GREEN='\033[0;32m'; _YELLOW='\033[1;33m'
_CYAN='\033[0;36m'; _BOLD='\033[1m'; _NC='\033[0m'

log_info()    { echo -e "${_CYAN}[INFO]${_NC}  $*"; }
log_ok()      { echo -e "${_GREEN}[OK]${_NC}    $*"; }
log_warn()    { echo -e "${_YELLOW}[WARN]${_NC}  $*" >&2; }
log_error()   { echo -e "${_RED}[ERROR]${_NC} $*" >&2; }
log_step()    { echo -e "\n${_BOLD}${_CYAN}════════════════════════════════════════${_NC}"; \
                echo -e "${_BOLD}${_CYAN}  $*${_NC}"; \
                echo -e "${_BOLD}${_CYAN}════════════════════════════════════════${_NC}"; }
log_skip()    { echo -e "${_YELLOW}[SKIP]${_NC}  $*"; }

die() { log_error "$*"; exit 1; }

# Check required file/directory and give a meaningful error.
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

# Run a command, or print it in dry-run mode.
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
DATA_ROOT="data"
WSI_DIR=""            # derived from DATA_ROOT unless overridden
RESULTS_DIR="results"
ENCODER="conch"
GPU=0
N_FOLDS=5
SEED=42
S1_CONFIG="configs/ovarian_conch_s1.yaml"
S2_CONFIG="configs/ovarian_conch_s2.yaml"
PATCH_SIZE=512
THUMBNAIL=2048
WORKERS=8

RESUME=0
SKIP_FEATURES=0
SKIP_SUPERPIXELS=0
SKIP_SPLITS=0
SKIP_TRAINING=0
SKIP_STAGE1=0
SKIP_STAGE2=0
SKIP_HEATMAPS=0
DRY_RUN=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root)        DATA_ROOT="$2";    shift 2 ;;
        --wsi_dir)          WSI_DIR="$2";      shift 2 ;;
        --results)          RESULTS_DIR="$2";  shift 2 ;;
        --encoder)          ENCODER="$2";      shift 2 ;;
        --gpu)              GPU="$2";          shift 2 ;;
        --n_folds)          N_FOLDS="$2";      shift 2 ;;
        --seed)             SEED="$2";         shift 2 ;;
        --s1_config)        S1_CONFIG="$2";    shift 2 ;;
        --s2_config)        S2_CONFIG="$2";    shift 2 ;;
        --patch_size)       PATCH_SIZE="$2";   shift 2 ;;
        --thumbnail)        THUMBNAIL="$2";    shift 2 ;;
        --workers)          WORKERS="$2";      shift 2 ;;
        --resume)           RESUME=1;          shift   ;;
        --skip-features)    SKIP_FEATURES=1;   shift   ;;
        --skip-superpixels) SKIP_SUPERPIXELS=1;shift   ;;
        --skip-splits)      SKIP_SPLITS=1;     shift   ;;
        --skip-training)    SKIP_TRAINING=1;   shift   ;;
        --skip-stage1)      SKIP_STAGE1=1;     shift   ;;
        --skip-stage2)      SKIP_STAGE2=1;     shift   ;;
        --skip-heatmaps)    SKIP_HEATMAPS=1;   shift   ;;
        --dry-run)          DRY_RUN=1;         shift   ;;
        -h|--help)          usage ;;
        *) die "Unknown option: $1  (run with --help for usage)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths and validate
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

[[ -z "${WSI_DIR}" ]] && WSI_DIR="${DATA_ROOT}/wsi"

EMBED_DIR="${DATA_ROOT}/embeddings"
SP_DIR="${DATA_ROOT}/superpixels"
SPLIT_DIR="${DATA_ROOT}/splits"
LABELS_CSV="${DATA_ROOT}/labels.csv"

S1_OUT="${RESULTS_DIR}/stage1"
S2_OUT="${RESULTS_DIR}/stage2"
HEATMAP_OUT="${RESULTS_DIR}/heatmaps"

export CUDA_VISIBLE_DEVICES="${GPU}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo -e "${_BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║          SMMILe End-to-End Pipeline                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${_NC}"
echo "  Data root    : ${DATA_ROOT}"
echo "  WSI dir      : ${WSI_DIR}"
echo "  Results      : ${RESULTS_DIR}"
echo "  Encoder      : ${ENCODER}"
echo "  GPU          : ${GPU}"
echo "  Folds        : ${N_FOLDS}"
echo "  Seed         : ${SEED}"
echo "  Stage1 cfg   : ${S1_CONFIG}"
echo "  Stage2 cfg   : ${S2_CONFIG}"
[[ "${DRY_RUN}"     -eq 1 ]] && echo "  *** DRY RUN — no commands will be executed ***"
[[ "${RESUME}"      -eq 1 ]] && echo "  *** RESUME  — completed steps will be skipped ***"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks: Python + GPU
# ---------------------------------------------------------------------------
log_info "Checking Python environment..."
python3 -c "import torch; print('  PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())" \
    || die "PyTorch not available. Activate the correct conda/venv environment first."

# ---------------------------------------------------------------------------
# STEP 1 — Feature extraction
# ---------------------------------------------------------------------------
log_step "STEP 1 / 7 — Feature extraction"

_should_skip_01=0
if [[ "${SKIP_FEATURES}" -eq 1 || "${RESUME}" -eq 1 ]]; then
    n_embed_dirs=$(find "${EMBED_DIR}" -name "coords.csv" 2>/dev/null | wc -l)
    if [[ "${SKIP_FEATURES}" -eq 1 ]]; then
        log_skip "Step 1 skipped (--skip-features)."
        _should_skip_01=1
    elif [[ "${RESUME}" -eq 1 && "${n_embed_dirs}" -gt 0 ]]; then
        log_skip "Step 1 skipped — ${n_embed_dirs} slide(s) already embedded (--resume)."
        _should_skip_01=1
    fi
fi

if [[ "${_should_skip_01}" -eq 0 ]]; then
    require_dir "${WSI_DIR}" \
        "Place your .tif WSI files in '${WSI_DIR}/' before running the pipeline."
    n_wsi=$(find "${WSI_DIR}" -maxdepth 1 -name "*.tif" | wc -l)
    [[ "${n_wsi}" -eq 0 ]] && die "No .tif files found in ${WSI_DIR}/."
    log_info "Found ${n_wsi} .tif WSI file(s) in ${WSI_DIR}."

    run_cmd python scripts/01_extract_features.py \
        --encoder_name  "${ENCODER}" \
        --wsi_dir       "${WSI_DIR}" \
        --output_dir    "${EMBED_DIR}" \
        --patch_size    "${PATCH_SIZE}" \
        --step_size     "${PATCH_SIZE}" \
        --device        "cuda:${GPU}"
    log_ok "Feature extraction complete → ${EMBED_DIR}"
fi

# ---------------------------------------------------------------------------
# STEP 2 — Superpixel generation
# ---------------------------------------------------------------------------
log_step "STEP 2 / 7 — Superpixel generation"

_should_skip_02=0
if [[ "${SKIP_SUPERPIXELS}" -eq 1 ]]; then
    log_skip "Step 2 skipped (--skip-superpixels)."
    _should_skip_02=1
elif [[ "${RESUME}" -eq 1 ]]; then
    n_sp=$(find "${SP_DIR}" -name "*.npy" 2>/dev/null | wc -l)
    if [[ "${n_sp}" -gt 0 ]]; then
        log_skip "Step 2 skipped — ${n_sp} superpixel file(s) found (--resume)."
        _should_skip_02=1
    fi
fi

if [[ "${_should_skip_02}" -eq 0 ]]; then
    require_dir "${EMBED_DIR}" \
        "Run step 1 (feature extraction) first, or use --skip-features to point to existing embeddings."
    n_embed=$(find "${EMBED_DIR}" -name "coords.csv" 2>/dev/null | wc -l)
    [[ "${n_embed}" -eq 0 ]] && die \
        "No coords.csv files found under ${EMBED_DIR}/. Complete step 1 first."

    run_cmd python scripts/02_generate_superpixels.py \
        --embedding_dir "${EMBED_DIR}" \
        --output_dir    "${SP_DIR}" \
        --patch_size    "${PATCH_SIZE}"
    log_ok "Superpixel generation complete → ${SP_DIR}"
fi

# ---------------------------------------------------------------------------
# STEP 3 — Split preparation
# ---------------------------------------------------------------------------
log_step "STEP 3 / 7 — Cross-validation split preparation"

_should_skip_03=0
if [[ "${SKIP_SPLITS}" -eq 1 ]]; then
    log_skip "Step 3 skipped (--skip-splits)."
    _should_skip_03=1
elif [[ "${RESUME}" -eq 1 ]]; then
    n_splits=$(find "${SPLIT_DIR}" -name "splits_*.csv" 2>/dev/null | wc -l)
    if [[ "${n_splits}" -ge "${N_FOLDS}" ]]; then
        log_skip "Step 3 skipped — ${n_splits} split file(s) found (--resume)."
        _should_skip_03=1
    fi
fi

if [[ "${_should_skip_03}" -eq 0 ]]; then
    require_file "${LABELS_CSV}" \
        "Create '${LABELS_CSV}' with columns 'slide_id,label' (labels: CC/EC/HGSC/LGSC/MC)."

    run_cmd python scripts/03_prepare_splits.py \
        --labels     "${LABELS_CSV}" \
        --output_dir "${SPLIT_DIR}" \
        --n_folds    "${N_FOLDS}" \
        --seed       "${SEED}"
    log_ok "Splits written → ${SPLIT_DIR}"
fi

# ---------------------------------------------------------------------------
# STEP 4 — Stage 1 training (all folds)
# ---------------------------------------------------------------------------
log_step "STEP 4 / 7 — Stage 1 training (${N_FOLDS} folds)"

_should_skip_04=0
if [[ "${SKIP_TRAINING}" -eq 1 || "${SKIP_STAGE1}" -eq 1 ]]; then
    log_skip "Step 4 skipped (--skip-training / --skip-stage1)."
    _should_skip_04=1
elif [[ "${RESUME}" -eq 1 ]]; then
    # Consider done if all fold checkpoints exist
    all_s1_done=1
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        [[ ! -f "${S1_OUT}/fold${fold}/best_model.pth" ]] && all_s1_done=0
    done
    if [[ "${all_s1_done}" -eq 1 ]]; then
        log_skip "Step 4 skipped — all Stage 1 checkpoints exist (--resume)."
        _should_skip_04=1
    fi
fi

if [[ "${_should_skip_04}" -eq 0 ]]; then
    require_file "${S1_CONFIG}" \
        "Create '${S1_CONFIG}' or point --s1_config to an existing config file."
    require_dir "${EMBED_DIR}" \
        "Run step 1 (feature extraction) first."
    require_dir "${SPLIT_DIR}" \
        "Run step 3 (split preparation) first."

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        # Resume at fold level: skip folds already complete
        if [[ "${RESUME}" -eq 1 && -f "${S1_OUT}/fold${fold}/best_model.pth" ]]; then
            log_skip "  Stage 1 fold ${fold} already has a checkpoint — skipping."
            continue
        fi
        log_info "  Stage 1 | Fold ${fold} / $((N_FOLDS - 1)) ..."
        run_cmd python scripts/train.py \
            --config     "${S1_CONFIG}" \
            --stage      1 \
            --fold       "${fold}" \
            --seed       "${SEED}" \
            --output_dir "${S1_OUT}" \
            --data_root  "${DATA_ROOT}"
    done
    log_ok "Stage 1 training complete → ${S1_OUT}"
fi

# ---------------------------------------------------------------------------
# STEP 5+6 — Stage 2 training + evaluation (all folds)
# ---------------------------------------------------------------------------
log_step "STEP 5+6 / 7 — Stage 2 training + evaluation (${N_FOLDS} folds)"

_should_skip_05=0
if [[ "${SKIP_TRAINING}" -eq 1 || "${SKIP_STAGE2}" -eq 1 ]]; then
    log_skip "Steps 5+6 skipped (--skip-training / --skip-stage2)."
    _should_skip_05=1
elif [[ "${RESUME}" -eq 1 ]]; then
    all_s2_done=1
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        [[ ! -f "${S2_OUT}/fold${fold}/best_model.pth" ]] && all_s2_done=0
    done
    if [[ "${all_s2_done}" -eq 1 ]]; then
        log_skip "Steps 5+6 skipped — all Stage 2 checkpoints exist (--resume)."
        _should_skip_05=1
    fi
fi

if [[ "${_should_skip_05}" -eq 0 ]]; then
    require_file "${S2_CONFIG}" \
        "Create '${S2_CONFIG}' or point --s2_config to an existing config file."

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        S1_CKPT="${S1_OUT}/fold${fold}/best_model.pth"

        # Verify Stage 1 checkpoint for this fold
        if [[ ! -f "${S1_CKPT}" ]]; then
            die "Stage 1 checkpoint missing for fold ${fold}: ${S1_CKPT}
     → Complete step 4 (Stage 1 training) for fold ${fold} first, or use --skip-stage1 to skip it."
        fi

        if [[ "${RESUME}" -eq 1 && -f "${S2_OUT}/fold${fold}/best_model.pth" ]]; then
            log_skip "  Stage 2 fold ${fold} already has a checkpoint — skipping."
            continue
        fi

        log_info "  Stage 2 | Fold ${fold} / $((N_FOLDS - 1)) | Stage1: ${S1_CKPT}"
        # train.py with --stage 2 automatically runs test-set evaluation
        # at the end of training and writes inst_predictions_fold{N}.csv
        run_cmd python scripts/train.py \
            --config       "${S2_CONFIG}" \
            --stage        2 \
            --fold         "${fold}" \
            --seed         "${SEED}" \
            --stage1_ckpt  "${S1_CKPT}" \
            --output_dir   "${S2_OUT}" \
            --data_root    "${DATA_ROOT}"
    done
    log_ok "Stage 2 training + evaluation complete → ${S2_OUT}"

    # Print aggregate cross-fold metrics
    python3 -c "
import json, glob, numpy as np, sys
files = sorted(glob.glob('${S2_OUT}/fold*/eval_metrics.json'))
if not files:
    print('  (no eval_metrics.json files found yet)')
    sys.exit(0)
keys = ['wsi_auc','patch_auc','patch_f1','patch_acc','patch_precision','patch_recall']
vals = {k: [] for k in keys}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    for k in keys:
        vals[k].append(d.get(k, float('nan')))
print('  ===== ${N_FOLDS}-fold cross-validation results (mean ± std) =====')
for k in keys:
    arr = np.array(vals[k])
    print(f'    {k:22s}: {arr.mean():.4f} ± {arr.std():.4f}')
"
fi

# ---------------------------------------------------------------------------
# STEP 7 — Heatmap generation
# ---------------------------------------------------------------------------
log_step "STEP 7 / 7 — Heatmap generation"

_should_skip_07=0
if [[ "${SKIP_HEATMAPS}" -eq 1 ]]; then
    log_skip "Step 7 skipped (--skip-heatmaps)."
    _should_skip_07=1
elif [[ "${RESUME}" -eq 1 ]]; then
    n_hm=$(find "${HEATMAP_OUT}" -name "*_heatmap.png" 2>/dev/null | wc -l)
    if [[ "${n_hm}" -gt 0 ]]; then
        log_skip "Step 7 skipped — ${n_hm} heatmap(s) already exist (--resume)."
        _should_skip_07=1
    fi
fi

if [[ "${_should_skip_07}" -eq 0 ]]; then
    # Verify at least one inst_predictions CSV exists
    n_pred_csv=$(find "${S2_OUT}" -name "inst_predictions_fold*.csv" 2>/dev/null | wc -l)
    if [[ "${n_pred_csv}" -eq 0 ]]; then
        die "No instance prediction CSVs found under ${S2_OUT}/.
     → Complete steps 5+6 (Stage 2 training + evaluation) first, or use --skip-stage2."
    fi
    log_info "Found ${n_pred_csv} prediction CSV(s) — generating heatmaps..."

    run_cmd python scripts/07_generate_heatmaps.py \
        --wsi_dir         "${WSI_DIR}" \
        --predictions_dir "${S2_OUT}" \
        --output_dir      "${HEATMAP_OUT}" \
        --thumbnail_size  "${THUMBNAIL}" \
        --num_workers     "${WORKERS}"
    log_ok "Heatmaps saved → ${HEATMAP_OUT}"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${_BOLD}${_GREEN}══════════════════════════════════════════════════${_NC}"
echo -e "${_BOLD}${_GREEN}  Pipeline complete!${_NC}"
echo -e "${_BOLD}${_GREEN}══════════════════════════════════════════════════${_NC}"
echo ""
echo "  Stage 1 checkpoints : ${S1_OUT}/fold*/best_model.pth"
echo "  Stage 2 checkpoints : ${S2_OUT}/fold*/best_model.pth"
echo "  Eval metrics (JSON) : ${S2_OUT}/fold*/eval_metrics.json"
echo "  Heatmaps            : ${HEATMAP_OUT}/*_heatmap.png"
echo ""
