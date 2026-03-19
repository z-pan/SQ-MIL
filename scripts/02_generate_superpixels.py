#!/usr/bin/env python3
"""
Script 02 — Generate SLIC superpixel maps aligned with patch coordinates.

For each slide the script:
  1. Loads the patch coordinate CSV (output of script 01).
  2. Re-arranges embeddings into a 2-D grid (NIC-style).
  3. Applies SLIC superpixel segmentation.
  4. Saves a 1-D label array aligned with the coordinate order as {slide_id}.npy.

Parameters (from paper):
  n_segments_persp = 25  (for 5×5 superpatch initial size)
  compactness = 50

Usage::

    python scripts/02_generate_superpixels.py \
        --embedding_dir data/embeddings \
        --output_dir data/superpixels \
        --n_segments 25 \
        --compactness 50
"""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import slic
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SLIC superpixel maps for MIL bags.")
    p.add_argument("--embedding_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--n_segments", type=int, default=25,
                   help="Target number of superpixels per superpatch (n_segments_persp).")
    p.add_argument("--compactness", type=float, default=50.0,
                   help="SLIC compactness parameter.")
    return p.parse_args()


def generate_superpixels_for_slide(
    slide_dir: Path,
    output_dir: Path,
    n_segments_persp: int,
    compactness: float,
) -> None:
    slide_id = slide_dir.name
    out_path = output_dir / f"{slide_id}.npy"

    if out_path.exists():
        logger.info("Skipping %s — superpixel map already exists.", slide_id)
        return

    coords_csv = slide_dir / "coords.csv"
    if not coords_csv.exists():
        logger.warning("No coords.csv for %s — skipping.", slide_id)
        return

    coords_df = pd.read_csv(coords_csv)
    n = len(coords_df)
    if n == 0:
        logger.warning("Empty coords.csv for %s — skipping.", slide_id)
        return

    # Load embeddings
    emb_list: list[np.ndarray] = []
    for _, row in coords_df.iterrows():
        x, y, ps = int(row["x"]), int(row["y"]), int(row["patch_size"])
        npy_path = slide_dir / f"{x}_{y}_{ps}.npy"
        if npy_path.exists():
            emb_list.append(np.load(npy_path))
        else:
            emb_list.append(np.zeros(512, dtype=np.float32))  # placeholder

    embeddings = np.stack(emb_list, axis=0)  # (N, D)

    # Arrange into 2-D grid for SLIC
    side = math.ceil(math.sqrt(n))
    pad_len = side * side - n
    padded = np.concatenate([embeddings, np.zeros((pad_len, embeddings.shape[1]), dtype=np.float32)], axis=0)
    grid = padded.reshape(side, side, -1)  # (side, side, D)

    # Normalise to [0, 1] for SLIC
    g_min, g_max = grid.min(), grid.max()
    if g_max > g_min:
        grid_norm = (grid - g_min) / (g_max - g_min)
    else:
        grid_norm = grid

    total_segments = n_segments_persp * (side * side // n_segments_persp + 1)
    segments = slic(
        grid_norm,
        n_segments=total_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )  # (side, side) int array

    # Flatten and remove padding
    sp_labels = segments.flatten()[:n]

    np.save(str(out_path), sp_labels)
    logger.info("%s: %d patches → %d superpixels.", slide_id, n, len(np.unique(sp_labels)))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slide_dirs = sorted([d for d in args.embedding_dir.iterdir() if d.is_dir()])
    if not slide_dirs:
        logger.error("No slide directories found in %s", args.embedding_dir)
        return

    logger.info("Processing %d slides.", len(slide_dirs))
    for slide_dir in tqdm(slide_dirs, desc="Superpixels"):
        generate_superpixels_for_slide(
            slide_dir=slide_dir,
            output_dir=args.output_dir,
            n_segments_persp=args.n_segments,
            compactness=args.compactness,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
