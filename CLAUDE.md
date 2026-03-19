# CLAUDE.md — SQ-MIL Project

## Project Overview

This project (**SQ-MIL**) implements the **SMMILe** (Superpatch-based Measurable Multiple Instance Learning) framework for spatial quantification of ovarian cancer whole-slide images (WSIs). The primary deliverable is per-case **attention heatmaps** showing spatial predictions overlaid on H&E-stained tissue.

- **Paper**: Gao et al., "SMMILe enables accurate spatial quantification in digital pathology using multiple-instance learning", *Nature Cancer* (2025). DOI: 10.1038/s43018-025-01060-8
- **Upstream repo**: https://github.com/ZeyuGaoAi/SMMILe (GPLv3)
- **This repo**: https://github.com/z-pan/SQ-MIL.git

## Critical Data Format

All WSI images are in **`.tif` format** (NOT `.svs`). This includes:
- UBC-OCEAN public dataset images (originally `.png`, converted to `.tif`)
- Our lab's private ovarian cancer H&E WSIs (native `.tif`)

The upstream SMMILe `generate_heatmap.py` uses `openslide.OpenSlide()` which supports `.tif`/`.tiff` (BigTIFF/TIFF via OpenSlide), but the glob pattern and file handling must be adapted from `*.svs` to `*.tif`.

## Task: Ovarian Cancer 5-Subtype Classification

- **Dataset**: UBC-OCEAN (513 WSIs) + custom lab data
- **Classes**: CC (clear cell), EC (endometrioid), HGSC (high-grade serous), LGSC (low-grade serous), MC (mucinous) — 5 subtypes
- **Task type**: Multiclass classification (use `single/` directory from SMMILe, NOT `multi/`)
- **Encoder**: Conch (pathology foundation model, 512-dim embeddings) preferred; ResNet-50 (ImageNet, 1024-dim) as fallback
- **Patch size**: 512×512 at 20× magnification, level=0

## Architecture Summary

SMMILe has two training stages:

### Stage 1 — Primary Network
- Frozen pretrained encoder → convolutional layer (3×3, NIC-based) → instance detector (gated attention, per-category) + instance classifier
- Parameter-free instance dropout (InD): drops high-scoring instances stochastically
- Delocalized instance sampling (InS): SLIC superpatch-based pseudo-bag generation
- Loss: BCE classification loss (L_cls)
- Epochs: 40 (Conch) or 200 (ResNet-50)

### Stage 2 — Instance Refinement + MRF
- Loads Stage 1 weights
- Adds N=3 refinement linear layers with progressive pseudo-labeling (top-θ% selection, θ=10%)
- Adds superpatch-based MRF constraint for spatial smoothness (λ1=0.8, λ2=0.2)
- Loss: L_cls + L_ref + L_mrf (+ L_cons if dataset has normal cases)
- Epochs: 20 (Conch) or 100 (ResNet-50)

### Inference & Heatmaps
- WSI-level: dot product of detection and classification scores → bag prediction
- Patch-level: last refinement layer v_N(·) outputs (C+1)-dim softmax → spatial prediction per patch
- Heatmap: overlay patch predictions on WSI thumbnail using colormap + Gaussian smoothing

## Directory Structure to Generate

```
SQ-MIL/
├── CLAUDE.md                          # This file
├── README.md
├── requirements.txt
├── configs/
│   ├── ovarian_conch_s1.yaml          # Stage 1 config
│   └── ovarian_conch_s2.yaml          # Stage 2 config
├── scripts/
│   ├── 01_extract_features.py         # Patch embedding extraction from .tif WSIs
│   ├── 02_generate_superpixels.py     # SLIC superpixel segmentation
│   ├── 03_prepare_splits.py           # 5-fold CV split generation
│   ├── 04_train_stage1.sh             # Stage 1 training shell script
│   ├── 05_train_stage2.sh             # Stage 2 training shell script
│   ├── 06_evaluate.sh                 # Evaluation shell script
│   └── 07_generate_heatmaps.py        # Heatmap generation (adapted for .tif)
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── smmile.py                  # SMMILe model definition
│   │   ├── attention.py               # Gated attention mechanism
│   │   ├── instance_refinement.py     # Instance refinement network
│   │   └── nic.py                     # Neural image compression layer
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── mil_dataset.py             # MIL dataset loader (embeddings + superpixels)
│   │   └── wsi_utils.py               # WSI reading utilities (.tif support)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training loop (Stage 1 + Stage 2)
│   │   ├── losses.py                  # All loss functions (L_cls, L_cons, L_ref, L_mrf)
│   │   └── evaluator.py              # Evaluation metrics
│   └── visualization/
│       ├── __init__.py
│       └── heatmap.py                 # Heatmap generation core (adapted for .tif)
├── data/                              # Data directory (not tracked in git)
│   ├── wsi/                           # Raw .tif WSI files
│   ├── embeddings/                    # Extracted patch embeddings (.npy)
│   ├── superpixels/                   # Superpixel segmentation results (.npy)
│   ├── splits/                        # Cross-validation split CSVs
│   └── labels.csv                     # slide_id, label columns
└── results/                           # Training results + heatmaps (not tracked)
    ├── stage1/
    ├── stage2/
    └── heatmaps/
```

## Key Implementation Requirements

### 1. .tif WSI Support (CRITICAL)
- All file glob patterns must use `*.tif` instead of `*.svs`
- `openslide.OpenSlide()` works with `.tif`/`.tiff` natively if they are tiled/pyramidal TIFF
- For non-pyramidal `.tif` files, fall back to `PIL.Image.open()` or `tifffile.imread()`
- The `wsi_utils.py` module must detect format and use appropriate reader
- Heatmap generation must handle both pyramidal (OpenSlide) and flat (PIL/tifffile) TIFF

### 2. Feature Extraction
- Must support both Conch and ResNet-50 encoders
- Conch: requires `timm` and HuggingFace model `conch_v1` (512-dim output)
- ResNet-50: third residual block output, ImageNet pretrained (1024-dim output)
- Tessellate WSI into non-overlapping 512×512 patches at 20×
- Skip background patches (Otsu thresholding on grayscale)
- Save embeddings as `.npy` files: `{slide_id}/{x}_{y}_{patch_size}.npy`
- Also save a coordinate CSV per slide for heatmap reconstruction

### 3. Superpixel Generation
- Use SLIC from `skimage.segmentation.slic` on compressed WSI (NIC-style rearrangement)
- n_segments_persp = 25 (for 5×5 superpatch initial size, appropriate for multiclass)
- compactness = 50
- Save as `.npy` per slide

### 4. Training
- 5-fold cross-validation, patient-level splits (80% train+val, 20% test; train:val = 90:10)
- Weighted sampling for class imbalance
- Adam optimizer, lr=2e-5
- Early stopping on validation loss
- UBC-OCEAN is multiclass (NOT multilabel), so use softmax for instance classifier + BCE per category for bag loss
- No `--consistency` flag for UBC-OCEAN (no explicit normal class — all WSIs have cancer subtypes)
- Conv kernel size = 3×3 (not 1×1; 1×1 only for Camelyon16)

### 5. Heatmap Generation (HIGHEST PRIORITY)
This is the most important deliverable. For each test WSI, produce:
- A `.png` image showing the WSI thumbnail with colored overlay
- Each patch colored by its predicted subtype (use distinct colors per class)
- For multiclass: use category-specific colors, not just red intensity
  - CC → color 1, EC → color 2, HGSC → color 3, LGSC → color 4, MC → color 5
- Apply Gaussian smoothing (radius=5) to the overlay mask
- Include ground truth heatmap side-by-side when annotations are available
- Save per-slide instance prediction CSV: filename, x, y, predicted_class, prob_per_class
- The script must accept `--wsi_dir '/path/*.tif'` (not *.svs)

Color scheme for ovarian subtypes (suggested):
```python
SUBTYPE_COLORS = {
    0: (230, 25, 75),    # CC - Red
    1: (60, 180, 75),    # EC - Green
    2: (255, 225, 25),   # HGSC - Yellow
    3: (0, 130, 200),    # LGSC - Blue
    4: (145, 30, 180),   # MC - Purple
}
```

### 6. Evaluation Metrics
- WSI-level: macro AUC (one-vs-rest)
- Patch-level spatial quantification: macro AUC, macro F1, accuracy, precision, recall
- Per-fold and mean±std across 5 folds

## Upstream Code Reference

The SMMILe original repo structure:
```
SMMILe/
├── single/           # Binary & multiclass classification (USE THIS for UBC-OCEAN)
│   ├── main.py       # Training entry point
│   ├── eval.py       # Evaluation entry point
│   ├── demo.py       # Single WSI demo
│   ├── metric_calculate.py
│   ├── models/       # Model definitions
│   ├── datasets/     # Dataset loaders
│   └── configs_*/    # YAML configs per dataset
├── multi/            # Multilabel classification (NOT for UBC-OCEAN)
├── feature_extraction.py      # From raw WSI
├── feature_extraction_patch.py # From pre-tessellated patches
├── superpixel_generation.py
├── generate_heatmap.py        # Heatmap overlay (uses openslide, expects .svs)
└── pre_utils.py               # Preprocessing utilities
```

Key upstream files to study and adapt:
- `single/main.py` — training loop with Stage 1/Stage 2 logic
- `single/models/` — SMMILe model architecture
- `generate_heatmap.py` — heatmap overlay (needs .tif adaptation)
- `feature_extraction.py` — embedding extraction pipeline
- `superpixel_generation.py` — SLIC superpixel generation

The upstream `generate_heatmap.py` expects `_inst.csv` result files with columns: `filename, prob, label, svs_name`. It uses `openslide.OpenSlide()` and creates RGBA overlays with `jet` colormap. For multiclass, this needs to be extended to show per-class colors instead of single probability heatmaps.

## Environment

- Python 3.10+
- PyTorch 1.12+ with CUDA 11.3+
- Key packages: openslide-python, scikit-image, scikit-learn, pandas, numpy, h5py, opencv-python, matplotlib, tqdm, pyyaml, tifffile, Pillow
- For Conch encoder: timm, transformers, huggingface_hub
- GPU: NVIDIA A100 (40GB) or similar

## HuggingFace Resources

- Pre-extracted embeddings (all 6 public datasets): `zeyugao/SMMILe_Datasets`
- Spatial annotations: `zeyugao/SMMILe_SpatialAnnotation`
- No pre-trained model weights are available — must train from scratch

## Paper Performance Reference (UBC-OCEAN)

| Metric | ResNet-50 | Conch |
|---|---|---|
| WSI AUC | 94.11% | 97.01% |
| Spatial AUC | 94.40% | 96.67% |
| Spatial F1 | 95.27% | 96.02% |
| Spatial Accuracy | 91.83% | 92.98% |
