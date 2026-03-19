#!/usr/bin/env python3
"""
Script 03 — Prepare 5-fold cross-validation splits for ovarian cancer MIL.

Strategy
--------
  • Stratified by label (CC / EC / HGSC / LGSC / MC) using StratifiedKFold.
  • Splits are at *patient* level: all slides from the same patient always
    land in the same fold (train / val / test).
    Patient ID is inferred from slide_id as the prefix before the first '_',
    or the full slide_id when no '_' is present.
  • Per fold:
        20 %  of patients  →  test
        80 %  of patients  →  train + val
          └─ 90 % of 80 %  →  train
          └─ 10 % of 80 %  →  val    (stratified sub-split)

Output
------
    <output_dir>/splits_0.csv
                 splits_1.csv
                 ...
                 splits_4.csv

Each CSV has three columns: ``train``, ``val``, ``test``.
Each column contains the slide_ids belonging to that subset.
Columns are NaN-padded to equal length (the longest subset).

Label mapping (for StratifiedKFold stratification)
---------------------------------------------------
  CC → 0  EC → 1  HGSC → 2  LGSC → 3  MC → 4

Usage
-----
    python scripts/03_prepare_splits.py \\
        --labels     data/labels.csv \\
        --output_dir data/splits \\
        --n_folds    5 \\
        --seed       42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label map (must match mil_dataset.LABEL_MAP)
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "CC":   0,
    "EC":   1,
    "HGSC": 2,
    "LGSC": 3,
    "MC":   4,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 5-fold stratified CV splits at patient level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--labels", type=Path, default=Path("data/labels.csv"),
        help="Input CSV with columns: slide_id, label.",
    )
    p.add_argument(
        "--output_dir", type=Path, default=Path("data/splits"),
        help="Directory where splits_0.csv … splits_{n_folds-1}.csv are written.",
    )
    p.add_argument("--n_folds",   type=int,   default=5)
    p.add_argument("--val_ratio", type=float, default=0.10,
                   help="Fraction of (train+val) patients to use as validation.")
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_patient_id(slide_id: str) -> str:
    """Extract patient ID from slide_id.

    Convention: the part before the first '_' is the patient identifier.
    If no '_' is present the whole slide_id is the patient ID.

    Examples
    --------
    >>> infer_patient_id("TCGA-01-A234_slide1")
    'TCGA-01-A234'
    >>> infer_patient_id("slide001")
    'slide001'
    """
    idx = slide_id.find("_")
    return slide_id[:idx] if idx != -1 else slide_id


def encode_labels(series: pd.Series) -> np.ndarray:
    """Convert a string-label Series to an integer numpy array.

    Accepts both string labels (CC / EC / HGSC / LGSC / MC) and integer
    labels already present in the CSV.
    """
    def _encode(v):
        if isinstance(v, str):
            v = v.strip()
            if v not in LABEL_MAP:
                raise ValueError(
                    f"Unknown label '{v}'. Valid labels: {sorted(LABEL_MAP)}."
                )
            return LABEL_MAP[v]
        return int(v)

    return np.array([_encode(v) for v in series], dtype=np.int64)


def _make_split_df(
    train_ids: list[str],
    val_ids:   list[str],
    test_ids:  list[str],
) -> pd.DataFrame:
    """Build a three-column DataFrame, NaN-padded to equal length."""
    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    return pd.DataFrame({
        "train": pd.array(train_ids + [pd.NA] * (max_len - len(train_ids)),
                          dtype=pd.StringDtype()),
        "val":   pd.array(val_ids   + [pd.NA] * (max_len - len(val_ids)),
                          dtype=pd.StringDtype()),
        "test":  pd.array(test_ids  + [pd.NA] * (max_len - len(test_ids)),
                          dtype=pd.StringDtype()),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    df = pd.read_csv(args.labels)
    df["slide_id"] = df["slide_id"].astype(str).str.strip()

    # Validate and encode labels
    try:
        df["label_int"] = encode_labels(df["label"])
    except ValueError as exc:
        logger.error("Label encoding failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------- patient-level grouping
    df["patient_id"] = df["slide_id"].apply(infer_patient_id)

    # One representative label per patient (majority vote; ties broken by min)
    patient_df = (
        df.groupby("patient_id")["label_int"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .reset_index()
        .rename(columns={"label_int": "patient_label"})
    )

    patients  = patient_df["patient_id"].values          # (P,) str
    p_labels  = patient_df["patient_label"].values       # (P,) int

    n_patients = len(patients)
    logger.info(
        "Dataset: %d slides, %d patients, %d classes",
        len(df), n_patients, len(np.unique(p_labels)),
    )

    # Per-class patient counts (useful for sanity-checking balance)
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    for lbl, cnt in zip(*np.unique(p_labels, return_counts=True)):
        logger.info("  %-6s (%d): %d patients",
                    inv_map.get(int(lbl), str(lbl)), int(lbl), int(cnt))

    # -------------------------------------------------- 5-fold outer split
    skf_outer = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )

    # Number of splits for the inner val split (e.g. val_ratio=0.1 → 10 splits)
    n_inner_splits = max(2, round(1.0 / args.val_ratio))
    skf_inner = StratifiedKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=args.seed
    )

    for fold, (trainval_idx, test_idx) in enumerate(
        skf_outer.split(patients, p_labels)
    ):
        test_patients    = set(patients[test_idx])
        trainval_patients = patients[trainval_idx]
        trainval_labels   = p_labels[trainval_idx]

        # Inner split: take the first fold of skf_inner as the val set
        train_sub_idx, val_sub_idx = next(
            skf_inner.split(trainval_patients, trainval_labels)
        )
        train_patients = set(trainval_patients[train_sub_idx])
        val_patients   = set(trainval_patients[val_sub_idx])

        # Map patients back to slides
        train_ids = (
            df[df["patient_id"].isin(train_patients)]["slide_id"]
            .sort_values().tolist()
        )
        val_ids = (
            df[df["patient_id"].isin(val_patients)]["slide_id"]
            .sort_values().tolist()
        )
        test_ids = (
            df[df["patient_id"].isin(test_patients)]["slide_id"]
            .sort_values().tolist()
        )

        # Sanity: no overlap between subsets
        assert not (set(train_ids) & set(val_ids)),  "train/val overlap!"
        assert not (set(train_ids) & set(test_ids)), "train/test overlap!"
        assert not (set(val_ids)   & set(test_ids)), "val/test overlap!"

        split_df = _make_split_df(train_ids, val_ids, test_ids)
        out_path = args.output_dir / f"splits_{fold}.csv"
        split_df.to_csv(out_path, index=False)

        # Label distribution of train/val/test slides
        def _dist(ids: list[str]) -> str:
            sub = df[df["slide_id"].isin(ids)]
            counts = sub["label_int"].value_counts().sort_index()
            return "  ".join(
                f"{inv_map.get(int(lbl), str(lbl))}:{int(cnt)}"
                for lbl, cnt in counts.items()
            )

        logger.info(
            "Fold %d — train: %3d slides (%2d pt) [%s]",
            fold, len(train_ids), len(train_patients), _dist(train_ids),
        )
        logger.info(
            "          val:   %3d slides (%2d pt) [%s]",
            len(val_ids), len(val_patients), _dist(val_ids),
        )
        logger.info(
            "          test:  %3d slides (%2d pt) [%s]",
            len(test_ids), len(test_patients), _dist(test_ids),
        )
        logger.info("          → %s", out_path)

    logger.info("All splits saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
