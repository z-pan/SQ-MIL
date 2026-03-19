"""
MIL bag dataset loader.

Each item is a bag (WSI) represented by:
  - A matrix of patch embeddings  (N, embedding_dim)
  - A superpixel label map         (N,)  [int, 0-indexed superpatch ID]
  - The bag-level class label      (int)

Embeddings are pre-extracted and stored as .npy files.
Superpixel maps are stored as .npy files (one per slide).

Directory layout::

    embeddings/
        {slide_id}/
            coords.csv           # columns: x, y, patch_size
            {x}_{y}_{size}.npy  # one file per patch (embedding vector)

    superpixels/
        {slide_id}.npy           # superpixel label array aligned with coords.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MILDataset(Dataset):
    """MIL bag dataset that reads pre-extracted patch embeddings.

    Args:
        slide_ids:      List of slide identifiers (matching embedding directory names).
        labels:         Dict mapping slide_id → int label.
        embedding_dir:  Root directory of pre-extracted embeddings.
        superpixel_dir: Root directory of superpixel .npy files.
        augment:        Whether to apply bag-level augmentation (random patch drop).
    """

    def __init__(
        self,
        slide_ids: list[str],
        labels: dict[str, int],
        embedding_dir: str | Path,
        superpixel_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self.slide_ids = slide_ids
        self.labels = labels
        self.embedding_dir = Path(embedding_dir)
        self.superpixel_dir = Path(superpixel_dir)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        slide_id = self.slide_ids[idx]
        label = self.labels[slide_id]

        embeddings, coords = self._load_embeddings(slide_id)
        superpixels = self._load_superpixels(slide_id, n_patches=len(embeddings))

        if self.augment:
            embeddings, superpixels, coords = self._random_patch_drop(
                embeddings, superpixels, coords
            )

        return {
            "slide_id": slide_id,
            "embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "superpixels": torch.tensor(superpixels, dtype=torch.long),
            "coords": torch.tensor(coords, dtype=torch.long),
            "label": label,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_embeddings(self, slide_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load all patch embeddings and coordinate array for a slide.

        Returns:
            embeddings: (N, embedding_dim)
            coords:     (N, 2) — (x, y) in level-0 pixel coordinates
        """
        slide_dir = self.embedding_dir / slide_id
        coords_csv = slide_dir / "coords.csv"

        if not coords_csv.exists():
            raise FileNotFoundError(
                f"Coordinate file not found: {coords_csv}. "
                "Run scripts/01_extract_features.py first."
            )

        coords_df = pd.read_csv(coords_csv)
        n = len(coords_df)
        emb_list: list[np.ndarray] = []
        coords_list: list[tuple[int, int]] = []

        for _, row in coords_df.iterrows():
            x, y, ps = int(row["x"]), int(row["y"]), int(row["patch_size"])
            npy_path = slide_dir / f"{x}_{y}_{ps}.npy"
            if not npy_path.exists():
                logger.warning("Missing embedding: %s — skipping", npy_path)
                continue
            emb_list.append(np.load(npy_path))
            coords_list.append((x, y))

        if not emb_list:
            raise RuntimeError(f"No valid embeddings found for slide {slide_id}")

        embeddings = np.stack(emb_list, axis=0)  # (N, D)
        coords = np.array(coords_list, dtype=np.int64)  # (N, 2)
        return embeddings, coords

    def _load_superpixels(self, slide_id: str, n_patches: int) -> np.ndarray:
        """Load superpixel label map aligned with patch order."""
        sp_path = self.superpixel_dir / f"{slide_id}.npy"
        if not sp_path.exists():
            logger.warning(
                "Superpixel file not found for %s — using trivial map.", slide_id
            )
            return np.arange(n_patches, dtype=np.int64)
        sp = np.load(sp_path)
        if len(sp) != n_patches:
            logger.warning(
                "Superpixel length mismatch for %s (%d vs %d) — truncating/padding.",
                slide_id,
                len(sp),
                n_patches,
            )
            if len(sp) > n_patches:
                sp = sp[:n_patches]
            else:
                sp = np.pad(sp, (0, n_patches - len(sp)), constant_values=sp[-1])
        return sp.astype(np.int64)

    @staticmethod
    def _random_patch_drop(
        embeddings: np.ndarray,
        superpixels: np.ndarray,
        coords: np.ndarray,
        drop_rate: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly drop a fraction of patches for bag-level augmentation."""
        n = len(embeddings)
        keep_n = max(1, int(n * (1.0 - drop_rate)))
        keep_idx = np.random.choice(n, keep_n, replace=False)
        keep_idx = np.sort(keep_idx)
        return embeddings[keep_idx], superpixels[keep_idx], coords[keep_idx]


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataset(
    split_csv: str | Path,
    labels_csv: str | Path,
    embedding_dir: str | Path,
    superpixel_dir: str | Path,
    augment: bool = False,
) -> MILDataset:
    """Build a MILDataset from a split CSV and labels CSV.

    Args:
        split_csv:   CSV with at least a 'slide_id' column.
        labels_csv:  CSV with columns 'slide_id' and 'label'.
        embedding_dir, superpixel_dir: Data directories.
        augment:     Enable data augmentation.
    """
    split_df = pd.read_csv(split_csv)
    labels_df = pd.read_csv(labels_csv)
    label_map: dict[str, int] = dict(
        zip(labels_df["slide_id"].astype(str), labels_df["label"].astype(int))
    )
    slide_ids = split_df["slide_id"].astype(str).tolist()
    # Filter to slides that have labels
    slide_ids = [s for s in slide_ids if s in label_map]
    return MILDataset(
        slide_ids=slide_ids,
        labels=label_map,
        embedding_dir=embedding_dir,
        superpixel_dir=superpixel_dir,
        augment=augment,
    )
