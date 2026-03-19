# SQ-MIL: Spatial Quantification via Multiple Instance Learning

Implementation of the **SMMILe** (Superpatch-based Measurable Multiple Instance Learning) framework for spatial quantification of ovarian cancer whole-slide images (WSIs), adapted for `.tif` format inputs and 5-subtype classification.

**Paper**: Gao et al., "SMMILe enables accurate spatial quantification in digital pathology using multiple-instance learning", *Nature Cancer* (2025). DOI: [10.1038/s43018-025-01060-8](https://doi.org/10.1038/s43018-025-01060-8)

**Upstream repo**: [ZeyuGaoAi/SMMILe](https://github.com/ZeyuGaoAi/SMMILe) (GPLv3)

---

## Task

5-subtype ovarian cancer classification on WSIs:

| Label | Class | Color |
|-------|-------|-------|
| 0 | CC — Clear Cell | Red |
| 1 | EC — Endometrioid | Green |
| 2 | HGSC — High-Grade Serous | Yellow |
| 3 | LGSC — Low-Grade Serous | Blue |
| 4 | MC — Mucinous | Purple |

**Dataset**: UBC-OCEAN (513 WSIs) + optional private lab data. All images in `.tif` format.

---

## Architecture

SMMILe uses two training stages:

**Stage 1 — Primary Network**
- Frozen pretrained encoder (Conch or ResNet-50) → 3×3 NIC convolutional layer → gated attention instance detector + instance classifier
- Instance Dropout (InD) + Delocalized Instance Sampling (InS) via SLIC superpatches
- Loss: BCE classification loss (L_cls)

**Stage 2 — Instance Refinement + MRF**
- Loads Stage 1 weights; adds N=3 refinement linear layers
- Progressive pseudo-labeling (top-θ=10% selection)
- Superpatch-based MRF smoothness constraint (λ1=0.8, λ2=0.2)
- Loss: L_cls + L_ref + L_mrf

**Inference**: Patch predictions from last refinement layer → spatial heatmaps overlaid on WSI thumbnail.

---

## Performance (UBC-OCEAN, from paper)

| Metric | ResNet-50 | Conch |
|--------|-----------|-------|
| WSI AUC | 94.11% | 97.01% |
| Spatial AUC | 94.40% | 96.67% |
| Spatial F1 | 95.27% | 96.02% |
| Spatial Accuracy | 91.83% | 92.98% |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/z-pan/SQ-MIL.git
cd SQ-MIL
```

### 2. Create a conda environment

```bash
conda create -n sqmil python=3.10
conda activate sqmil
```

### 3. Install PyTorch

Install PyTorch matching your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command. Example for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Install OpenSlide system library

```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools libvips-dev

# macOS (via Homebrew)
brew install openslide
```

### 6. (Optional) Conch encoder setup

The Conch foundation model requires HuggingFace access. Log in and request access to `MahmoodLab/conch`:

```bash
huggingface-cli login
```

---

## Usage

### Step 1 — Extract patch features

```bash
python scripts/01_extract_features.py \
    --wsi_dir data/wsi \
    --output_dir data/embeddings \
    --encoder conch \
    --patch_size 512 \
    --mag 20
```

Supported encoders: `conch` (512-dim), `resnet50` (1024-dim).

### Step 2 — Generate superpixels

```bash
python scripts/02_generate_superpixels.py \
    --embedding_dir data/embeddings \
    --output_dir data/superpixels \
    --n_segments 25 \
    --compactness 50
```

### Step 3 — Prepare cross-validation splits

```bash
python scripts/03_prepare_splits.py \
    --labels data/labels.csv \
    --output_dir data/splits \
    --n_folds 5 \
    --seed 42
```

`labels.csv` must have columns: `slide_id`, `label` (0–4 integer).

### Step 4 — Train Stage 1

```bash
bash scripts/04_train_stage1.sh \
    --config configs/ovarian_conch_s1.yaml \
    --fold 0
```

Or train all folds:

```bash
for fold in 0 1 2 3 4; do
    bash scripts/04_train_stage1.sh --config configs/ovarian_conch_s1.yaml --fold $fold
done
```

### Step 5 — Train Stage 2

```bash
bash scripts/05_train_stage2.sh \
    --config configs/ovarian_conch_s2.yaml \
    --fold 0
```

### Step 6 — Evaluate

```bash
bash scripts/06_evaluate.sh \
    --config configs/ovarian_conch_s2.yaml \
    --fold 0
```

### Step 7 — Generate heatmaps

```bash
python scripts/07_generate_heatmaps.py \
    --wsi_dir 'data/wsi/*.tif' \
    --result_dir results/stage2/fold0 \
    --output_dir results/heatmaps \
    --labels data/labels.csv
```

Output per WSI:
- `{slide_id}_heatmap.png` — thumbnail with colored subtype overlay
- `{slide_id}_inst_pred.csv` — per-patch predictions (x, y, predicted_class, prob_CC, prob_EC, …)

---

## Directory Structure

```
SQ-MIL/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   ├── ovarian_conch_s1.yaml      # Stage 1 config (Conch encoder)
│   └── ovarian_conch_s2.yaml      # Stage 2 config (Conch encoder)
├── scripts/
│   ├── 01_extract_features.py
│   ├── 02_generate_superpixels.py
│   ├── 03_prepare_splits.py
│   ├── 04_train_stage1.sh
│   ├── 05_train_stage2.sh
│   ├── 06_evaluate.sh
│   └── 07_generate_heatmaps.py
├── src/
│   ├── models/
│   │   ├── smmile.py              # SMMILe model definition
│   │   ├── attention.py           # Gated attention mechanism
│   │   ├── instance_refinement.py # Instance refinement network
│   │   └── nic.py                 # NIC convolutional layer
│   ├── datasets/
│   │   ├── mil_dataset.py         # MIL bag dataset loader
│   │   └── wsi_utils.py           # .tif WSI reader (OpenSlide + PIL fallback)
│   ├── training/
│   │   ├── trainer.py             # Stage 1 & 2 training loops
│   │   ├── losses.py              # L_cls, L_ref, L_mrf, L_cons
│   │   └── evaluator.py           # AUC, F1, accuracy metrics
│   └── visualization/
│       └── heatmap.py             # Heatmap overlay generation
├── data/                          # Not tracked in git
│   ├── wsi/                       # Raw .tif WSI files
│   ├── embeddings/                # Patch embeddings (.npy)
│   ├── superpixels/               # SLIC superpixel maps (.npy)
│   ├── splits/                    # CV split CSVs
│   └── labels.csv
└── results/                       # Not tracked in git
    ├── stage1/
    ├── stage2/
    └── heatmaps/
```

---

## Data Format Notes

- **WSI format**: All images must be `.tif`. Both pyramidal TIFF (OpenSlide) and flat TIFF (PIL/tifffile fallback) are supported.
- **Embeddings**: Saved as `data/embeddings/{slide_id}/{x}_{y}_{patch_size}.npy` plus a `coords.csv` per slide.
- **Labels**: `data/labels.csv` with columns `slide_id` (string, no extension) and `label` (int 0–4).
- **Splits**: `data/splits/fold{k}_train.csv`, `fold{k}_val.csv`, `fold{k}_test.csv`.

---

## License

This project is derived from [SMMILe](https://github.com/ZeyuGaoAi/SMMILe) which is licensed under **GPLv3**. This repository is also released under GPLv3.
