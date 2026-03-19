#!/usr/bin/env python3
"""
Script 02 — Generate SLIC superpixel maps aligned with patch coordinates.

For each slide the script:
  1. Reads coords.csv produced by script 01 (columns: x, y, patch_size).
  2. Loads the corresponding patch embeddings (*.npy per patch).
  3. Places each embedding at its true spatial position in a 2-D grid
     ("NIC-style compressed WSI"):
       grid[row, col] = embedding   where row = y // patch_size
                                          col = x // patch_size
     Background cells (no tissue) remain zero-filled.
  4. Normalises the grid to [0, 1] and runs SLIC superpixel segmentation.
  5. Reads back the segment label for every tissue patch position.
  6. Re-indexes labels to contiguous 0..K-1 and saves as {slide_id}.npy
     in coords.csv row order (matching the embedding load order).

Parameters (from SMMILe paper):
  n_segments_persp = 25  → target ~25 tissue patches per superpatch (5×5 grid)
  compactness = 50

Why spatial placement matters
------------------------------
The superpixel labels feed two components of SMMILe Stage 2:
  • MRF loss  — penalises disagreement between patches sharing a superpatch;
                only meaningful if the superpatch groups spatially close patches.
  • InS (delocalized instance sampling) — pseudo-bag construction respects
                spatial proximity encoded in superpatch membership.

Flat enumeration ordering would cluster patches arbitrarily, destroying the
spatial signal.

Usage
-----
    python scripts/02_generate_superpixels.py \\
        --embedding_dir data/embeddings \\
        --output_dir    data/superpixels \\
        --patch_size    512 \\
        --n_segments_persp 25 \\
        --compactness   50
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import slic
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate SLIC superpixel maps for SMMILe MIL bags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embedding_dir", type=Path, required=True,
                   help="Root directory produced by script 01 "
                        "(contains one sub-dir per slide).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directory where {slide_id}.npy files are written.")
    p.add_argument("--patch_size", type=int, default=512,
                   help="Patch edge length in level-0 pixels "
                        "(must match script 01 --patch_size).")
    p.add_argument("--n_segments_persp", type=int, default=25,
                   help="Target number of tissue patches per superpatch "
                        "(paper default: 25 for 5×5 multiclass setting).")
    p.add_argument("--compactness", type=float, default=50.0,
                   help="SLIC compactness — higher values weight spatial "
                        "proximity more than feature similarity.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core per-slide logic
# ---------------------------------------------------------------------------

def _load_embeddings(coords_df: pd.DataFrame, slide_dir: Path) -> np.ndarray:
    """Load per-patch .npy embeddings in coords_df row order.

    Missing files are replaced with a zero vector of the same dimension as
    the first successfully loaded embedding.  If no embeddings can be loaded
    at all the function returns an empty (0, 1) array so callers can detect
    the degenerate case.
    """
    emb_list: list[np.ndarray | None] = []
    dim: int | None = None

    for _, row in coords_df.iterrows():
        x, y, ps = int(row["x"]), int(row["y"]), int(row["patch_size"])
        npy = slide_dir / f"{x}_{y}_{ps}.npy"
        if npy.exists():
            vec = np.load(str(npy)).astype(np.float32)
            if dim is None:
                dim = vec.shape[0]
            emb_list.append(vec)
        else:
            emb_list.append(None)

    if dim is None:
        return np.empty((0, 1), dtype=np.float32)

    # Replace missing entries with zeros
    filled = [v if v is not None else np.zeros(dim, dtype=np.float32)
              for v in emb_list]
    return np.stack(filled, axis=0)  # (N, D)


def _build_spatial_grid(
    embeddings: np.ndarray,
    grid_cols: np.ndarray,
    grid_rows: np.ndarray,
) -> np.ndarray:
    """Place embeddings into a 2-D spatial grid.

    Parameters
    ----------
    embeddings:  (N, D) float32 array.
    grid_cols:   (N,) column index for each patch  (= x // patch_size).
    grid_rows:   (N,) row    index for each patch  (= y // patch_size).

    Returns
    -------
    grid: np.ndarray of shape (max_row+1, max_col+1, D), float32.
          Background cells (no tissue) contain zeros.
    """
    n_rows = int(grid_rows.max()) + 1
    n_cols = int(grid_cols.max()) + 1
    D = embeddings.shape[1]

    grid = np.zeros((n_rows, n_cols, D), dtype=np.float32)
    for i in range(len(embeddings)):
        grid[grid_rows[i], grid_cols[i]] = embeddings[i]

    return grid


def _normalise_grid(grid: np.ndarray) -> np.ndarray:
    """Global min-max normalisation to [0, 1] across all channels and cells."""
    g_min = grid.min()
    g_max = grid.max()
    if g_max > g_min:
        return (grid - g_min) / (g_max - g_min)
    return grid.copy()


def generate_superpixels_for_slide(
    slide_dir: Path,
    output_dir: Path,
    patch_size: int,
    n_segments_persp: int,
    compactness: float,
) -> int:
    """Compute and save the superpixel label map for one slide.

    Returns the number of unique superpatches (0 on skip / failure).
    """
    slide_id = slide_dir.name
    out_path = output_dir / f"{slide_id}.npy"

    if out_path.exists():
        sp = np.load(str(out_path))
        logger.info("SKIP  %s — %s exists (%d unique superpatches).",
                    slide_id, out_path.name, len(np.unique(sp)))
        return int(len(np.unique(sp)))

    # ------------------------------------------------------------------ load
    coords_csv = slide_dir / "coords.csv"
    if not coords_csv.exists():
        logger.warning("SKIP  %s — coords.csv not found.", slide_id)
        return 0

    coords_df = pd.read_csv(coords_csv)
    n = len(coords_df)
    if n == 0:
        logger.warning("SKIP  %s — coords.csv is empty.", slide_id)
        return 0

    embeddings = _load_embeddings(coords_df, slide_dir)
    if embeddings.shape[0] == 0:
        logger.warning("SKIP  %s — no embedding .npy files found.", slide_id)
        return 0

    D = embeddings.shape[1]
    logger.debug("%s: %d patches, embedding dim=%d", slide_id, n, D)

    # ---------------------------------------- map (x,y) → spatial grid index
    xs = coords_df["x"].values.astype(np.int64)
    ys = coords_df["y"].values.astype(np.int64)
    gcols = xs // patch_size   # column index in compressed WSI grid
    grows = ys // patch_size   # row    index

    # --------------------------------------------------------- build 2-D grid
    grid = _build_spatial_grid(embeddings, gcols, grows)
    # grid shape: (n_grid_rows, n_grid_cols, D)
    n_grid_cells = grid.shape[0] * grid.shape[1]

    grid_norm = _normalise_grid(grid)

    # -------------------------------------------------- choose target segments
    # Target ~n_segments_persp tissue patches per superpatch.
    # Base on tissue count N, not on the full (sparse) grid size.
    n_segments_target = max(1, math.ceil(n / n_segments_persp))

    logger.debug(
        "%s: grid=%dx%d (%d cells), tissue=%d, target segments=%d",
        slide_id, grid.shape[0], grid.shape[1], n_grid_cells,
        n, n_segments_target,
    )

    # ------------------------------------------------------------------- SLIC
    # channel_axis=-1 tells skimage that the last dimension holds features.
    segments = slic(
        grid_norm,
        n_segments=n_segments_target,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )
    # segments: (n_grid_rows, n_grid_cols) integer array

    # -------------------------------------------- extract tissue patch labels
    # Read the SLIC label at each tissue patch's spatial position.
    sp_raw = np.array(
        [segments[grows[i], gcols[i]] for i in range(n)],
        dtype=np.int64,
    )

    # Re-index to contiguous 0..K-1 (background cells may consume some IDs
    # that no tissue patch actually received).
    _, sp_labels = np.unique(sp_raw, return_inverse=True)
    sp_labels = sp_labels.astype(np.int64)

    # ------------------------------------------------------------------ save
    np.save(str(out_path), sp_labels)
    n_unique = int(len(np.unique(sp_labels)))
    avg_per_sp = n / n_unique if n_unique > 0 else 0
    logger.info(
        "DONE  %s — %d patches → %d superpatches (avg %.1f patches/sp).",
        slide_id, n, n_unique, avg_per_sp,
    )
    return n_unique


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slide_dirs = sorted(d for d in args.embedding_dir.iterdir() if d.is_dir())
    if not slide_dirs:
        logger.error("No slide directories found in %s", args.embedding_dir)
        return

    logger.info(
        "Generating superpixels for %d slides  "
        "(patch_size=%d, n_segments_persp=%d, compactness=%.1f).",
        len(slide_dirs), args.patch_size, args.n_segments_persp, args.compactness,
    )

    total_sp = 0
    failed = 0
    for slide_dir in tqdm(slide_dirs, desc="Superpixels", unit="slide"):
        try:
            k = generate_superpixels_for_slide(
                slide_dir=slide_dir,
                output_dir=args.output_dir,
                patch_size=args.patch_size,
                n_segments_persp=args.n_segments_persp,
                compactness=args.compactness,
            )
            total_sp += k
        except Exception:
            logger.exception("Failed to process %s.", slide_dir.name)
            failed += 1

    logger.info(
        "All done.  Total unique superpatches across all slides: %d  "
        "(failures: %d).",
        total_sp, failed,
    )


if __name__ == "__main__":
    main()
