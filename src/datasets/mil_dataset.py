"""
MIL bag dataset loader for ovarian cancer subtype classification.

Each item is a bag (WSI) represented by:
  • patch embeddings matrix  (N, embedding_dim)  float32
  • superpixel label map     (N,)                int64  — 0-indexed superpatch ID
  • bag-level class label    scalar int
  • patch coordinates        (N, 2)              int64  — (x, y) in level-0 pixels
  • slide_id                 str

Label mapping (5-subtype UBC-OCEAN task)
-----------------------------------------
  CC   → 0    (clear cell carcinoma)
  EC   → 1    (endometrioid carcinoma)
  HGSC → 2    (high-grade serous carcinoma)
  LGSC → 3    (low-grade serous carcinoma)
  MC   → 4    (mucinous carcinoma)

Embeddings layout on disk
--------------------------
  <embedding_dir>/
      <slide_id>/
          coords.csv           columns: x, y, patch_size
          <x>_<y>_<size>.npy  one float32 vector per tissue patch

Superpixel layout
-----------------
  <superpixel_dir>/
      <slide_id>.npy           int64 (N,) — label per patch in coords.csv order

Split CSV format (produced by scripts/03_prepare_splits.py)
-------------------------------------------------------------
  splits_0.csv … splits_4.csv  — three columns: train, val, test
  Each column contains slide_ids for that subset; shorter columns are NaN-padded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "CC":   0,
    "EC":   1,
    "HGSC": 2,
    "LGSC": 3,
    "MC":   4,
}

# Inverse mapping for display / evaluation
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

NUM_CLASSES: int = len(LABEL_MAP)

SplitName = Literal["train", "val", "test"]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MILDataset(Dataset):
    """MIL bag dataset that reads pre-extracted patch embeddings.

    Parameters
    ----------
    slide_ids:
        Ordered list of slide identifiers.  Each entry must match a
        sub-directory inside *embedding_dir* and an entry in *labels*.
    labels:
        Dict ``{slide_id: int_label}`` for every slide in *slide_ids*.
    embedding_dir:
        Root directory of pre-extracted embeddings (script 01 output).
    superpixel_dir:
        Root directory of superpixel .npy files (script 02 output).
    augment:
        When True, randomly drops 10 % of patches per bag at each
        ``__getitem__`` call (bag-level augmentation for training).
    """

    def __init__(
        self,
        slide_ids: list[str],
        labels: dict[str, int],
        embedding_dir: str | Path,
        superpixel_dir: str | Path,
        augment: bool = False,
    ) -> None:
        self.slide_ids = list(slide_ids)
        self.labels = dict(labels)
        self.embedding_dir = Path(embedding_dir)
        self.superpixel_dir = Path(superpixel_dir)
        self.augment = augment

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return one bag.

        Returns
        -------
        dict with keys:
          ``slide_id``    str
          ``embeddings``  FloatTensor (N, D)
          ``superpixels`` LongTensor  (N,)
          ``coords``      LongTensor  (N, 2)  — (x, y) in level-0 pixels
          ``label``       int
        """
        slide_id = self.slide_ids[idx]
        label    = self.labels[slide_id]

        embeddings, coords = self._load_embeddings(slide_id)
        superpixels = self._load_superpixels(slide_id, n_patches=len(embeddings))

        if self.augment:
            embeddings, superpixels, coords = _random_patch_drop(
                embeddings, superpixels, coords
            )

        return {
            "slide_id":   slide_id,
            "embeddings": torch.tensor(embeddings,  dtype=torch.float32),
            "superpixels":torch.tensor(superpixels, dtype=torch.long),
            "coords":     torch.tensor(coords,      dtype=torch.long),
            "label":      label,
        }

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_embeddings(
        self, slide_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embeddings, coords)`` arrays for *slide_id*.

        embeddings : (N, D) float32
        coords     : (N, 2) int64   — columns are (x, y)
        """
        slide_dir  = self.embedding_dir / slide_id
        coords_csv = slide_dir / "coords.csv"

        if not coords_csv.exists():
            raise FileNotFoundError(
                f"Coordinate file not found: {coords_csv}. "
                "Run scripts/01_extract_features.py first."
            )

        coords_df = pd.read_csv(coords_csv)
        emb_list:    list[np.ndarray]    = []
        coords_list: list[tuple[int,int]] = []

        for _, row in coords_df.iterrows():
            x, y, ps = int(row["x"]), int(row["y"]), int(row["patch_size"])
            npy_path = slide_dir / f"{x}_{y}_{ps}.npy"
            if not npy_path.exists():
                logger.warning("Missing embedding: %s — skipping.", npy_path.name)
                continue
            emb_list.append(np.load(str(npy_path)).astype(np.float32))
            coords_list.append((x, y))

        if not emb_list:
            raise RuntimeError(
                f"No valid embeddings found for slide '{slide_id}'. "
                "Check that script 01 completed successfully."
            )

        return (
            np.stack(emb_list, axis=0),                     # (N, D) float32
            np.array(coords_list, dtype=np.int64),          # (N, 2) int64
        )

    def _load_superpixels(
        self, slide_id: str, n_patches: int
    ) -> np.ndarray:
        """Return superpixel label array aligned with *_load_embeddings* order.

        Falls back to a trivial identity map (every patch its own superpatch)
        when the .npy file is missing, with a logged warning.
        """
        sp_path = self.superpixel_dir / f"{slide_id}.npy"

        if not sp_path.exists():
            logger.warning(
                "Superpixel file not found for '%s' — "
                "using trivial identity map (each patch = one superpatch).",
                slide_id,
            )
            return np.arange(n_patches, dtype=np.int64)

        sp = np.load(str(sp_path))

        if len(sp) != n_patches:
            logger.warning(
                "Superpixel length mismatch for '%s': expected %d, got %d. "
                "Truncating or padding with last label.",
                slide_id, n_patches, len(sp),
            )
            if len(sp) > n_patches:
                sp = sp[:n_patches]
            else:
                sp = np.pad(sp, (0, n_patches - len(sp)),
                            mode="edge")

        return sp.astype(np.int64)


# ---------------------------------------------------------------------------
# Augmentation helper (module-level so it can be tested independently)
# ---------------------------------------------------------------------------

def _random_patch_drop(
    embeddings:  np.ndarray,
    superpixels: np.ndarray,
    coords:      np.ndarray,
    drop_rate:   float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly drop *drop_rate* fraction of patches (without replacement)."""
    n      = len(embeddings)
    keep_n = max(1, int(n * (1.0 - drop_rate)))
    idx    = np.sort(np.random.choice(n, keep_n, replace=False))
    return embeddings[idx], superpixels[idx], coords[idx]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def load_labels(labels_csv: str | Path) -> dict[str, int]:
    """Read labels.csv and return ``{slide_id: int_label}`` dict.

    Accepts both string labels (CC, EC, HGSC, LGSC, MC) and integer labels.
    String labels are mapped via ``LABEL_MAP``; integer labels are used as-is.

    Raises ``ValueError`` for unknown string labels.
    """
    df = pd.read_csv(labels_csv)
    df["slide_id"] = df["slide_id"].astype(str)

    result: dict[str, int] = {}
    for _, row in df.iterrows():
        sid = str(row["slide_id"])
        raw = row["label"]

        if isinstance(raw, str):
            raw = raw.strip()
            if raw not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label '{raw}' for slide '{sid}'. "
                    f"Valid labels: {sorted(LABEL_MAP)}."
                )
            result[sid] = LABEL_MAP[raw]
        else:
            result[sid] = int(raw)

    return result


def load_split_ids(
    split_csv: str | Path,
    split: SplitName,
) -> list[str]:
    """Read slide IDs for *split* ('train', 'val', or 'test') from a splits_N.csv.

    The CSV has three columns (train / val / test) with slide_ids;
    shorter columns are NaN-padded.  Returns slide IDs for the requested
    split, dropping NaN entries.
    """
    df = pd.read_csv(split_csv)
    if split not in df.columns:
        raise ValueError(
            f"Column '{split}' not found in {split_csv}. "
            f"Available columns: {list(df.columns)}."
        )
    return df[split].dropna().astype(str).tolist()


def build_dataset(
    split_csv:     str | Path,
    labels_csv:    str | Path,
    embedding_dir: str | Path,
    superpixel_dir:str | Path,
    split:         SplitName,
    augment:       bool = False,
) -> MILDataset:
    """Build a :class:`MILDataset` for a specific train/val/test fold.

    Parameters
    ----------
    split_csv:
        Path to ``splits_N.csv`` produced by ``scripts/03_prepare_splits.py``.
        Must have columns ``train``, ``val``, ``test``.
    labels_csv:
        Path to ``labels.csv`` with columns ``slide_id``, ``label``.
        Labels may be strings (CC/EC/HGSC/LGSC/MC) or integers.
    embedding_dir:
        Root directory of pre-extracted embeddings (script 01 output).
    superpixel_dir:
        Root directory of superpixel .npy files (script 02 output).
    split:
        Which subset to load: ``'train'``, ``'val'``, or ``'test'``.
    augment:
        Enable random patch-drop augmentation (recommended for training only).

    Returns
    -------
    MILDataset ready for use with a :class:`torch.utils.data.DataLoader`.

    Example
    -------
    >>> ds_train = build_dataset(
    ...     "data/splits/splits_0.csv",
    ...     "data/labels.csv",
    ...     "data/embeddings",
    ...     "data/superpixels",
    ...     split="train",
    ...     augment=True,
    ... )
    """
    label_map = load_labels(labels_csv)
    slide_ids = load_split_ids(split_csv, split)

    # Keep only slides that appear in labels_csv (guard against stale splits)
    missing = [s for s in slide_ids if s not in label_map]
    if missing:
        logger.warning(
            "%d slide(s) in '%s' split are absent from labels.csv — "
            "they will be dropped: %s",
            len(missing), split, missing[:10],
        )
    slide_ids = [s for s in slide_ids if s in label_map]

    if not slide_ids:
        raise RuntimeError(
            f"No valid slides for split='{split}' in {split_csv}. "
            "Check that labels.csv contains the expected slide IDs."
        )

    logger.info(
        "Built MILDataset: split=%s, n_slides=%d, augment=%s",
        split, len(slide_ids), augment,
    )
    return MILDataset(
        slide_ids=slide_ids,
        labels=label_map,
        embedding_dir=embedding_dir,
        superpixel_dir=superpixel_dir,
        augment=augment,
    )
