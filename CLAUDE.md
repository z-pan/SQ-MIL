# CLAUDE.md — SQ-MIL Project

## Project Overview

This project (**SQ-MIL**) implements the **SMMILe** (Superpatch-based Measurable Multiple Instance Learning) framework for spatial quantification of ovarian cancer whole-slide images (WSIs). The primary deliverable is per-case **attention heatmaps** showing spatial predictions overlaid on H&E-stained tissue.

- **Paper**: Gao et al., "SMMILe enables accurate spatial quantification in digital pathology using multiple-instance learning", *Nature Cancer* (2025). DOI: 10.1038/s43018-025-01060-8
- **Upstream repo**: https://github.com/ZeyuGaoAi/SMMILe (GPLv3)
- **This repo**: https://github.com/z-pan/SQ-MIL.git

## Commands

```bash
# Feature extraction from .tif WSIs
python scripts/01_extract_features.py --wsi_dir data/wsi --output_dir data/embeddings

# Superpixel generation
python scripts/02_generate_superpixels.py --wsi_dir data/wsi --output_dir data/superpixels

# Train Stage 1
bash scripts/04_train_stage1.sh

# Train Stage 2
bash scripts/05_train_stage2.sh

# Generate heatmaps (HIGHEST PRIORITY deliverable)
python scripts/07_generate_heatmaps.py --wsi_dir 'data/wsi/*.tif' --output_dir results/heatmaps
```

## Critical Data Format

All WSI images are in **`.tif` format** (NOT `.svs`). This includes:
- UBC-OCEAN public dataset images (originally `.png`, converted to `.tif`)
- Our lab's private ovarian cancer H&E WSIs (native `.tif`)

The upstream SMMILe `generate_heatmap.py` uses `openslide.OpenSlide()` which supports `.tif`/`.tiff` (BigTIFF/TIFF via OpenSlide), but the glob pattern and file handling must be adapted from `*.svs` to `*.tif`.

## Task

- **Dataset**: UBC-OCEAN (513 WSIs) + custom lab data
- **Classes**: CC (clear cell), EC (endometrioid), HGSC (high-grade serous), LGSC (low-grade serous), MC (mucinous) — 5 subtypes
- **Task type**: Multiclass classification (use `single/` directory from SMMILe, NOT `multi/`)
- **Encoder**: Conch (pathology foundation model, 512-dim embeddings) preferred; ResNet-50 (ImageNet, 1024-dim) as fallback
- **Patch size**: 512×512 at 20× magnification, level=0

## Architecture

Stage 1 — Primary Network: frozen encoder → convolutional layer (3×3, NIC-based) → gated attention instance detector + instance classifier. Uses instance dropout (InD) and SLIC superpatch-based pseudo-bag generation (InS). 40 epochs (Conch) / 200 epochs (ResNet-50).

Stage 2 — Instance Refinement: loads Stage 1 weights, adds N=3 refinement layers with progressive pseudo-labeling (θ=10%), MRF spatial smoothness constraint (λ1=0.8, λ2=0.2). 20 epochs (Conch) / 100 epochs (ResNet-50).

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

## Upstream Reference

Key files to study and adapt from the SMMILe upstream repo (https://github.com/ZeyuGaoAi/SMMILe):
- `single/main.py` — training loop with Stage 1/Stage 2 logic
- `single/models/` — SMMILe model architecture
- `generate_heatmap.py` — heatmap overlay (needs .tif adaptation, currently expects .svs)
- `feature_extraction.py` — embedding extraction pipeline
- `superpixel_generation.py` — SLIC superpixel generation

Use the `single/` directory, NOT `multi/`. UBC-OCEAN is multiclass, not multilabel.

## Environment

- Python 3.10+
- PyTorch 1.12+ with CUDA 11.3+
- Key packages: openslide-python, scikit-image, scikit-learn, pandas, numpy, h5py, opencv-python, matplotlib, tqdm, pyyaml, tifffile, Pillow
- For Conch encoder: timm, transformers, huggingface_hub
- GPU: NVIDIA A100 (40GB) or similar
