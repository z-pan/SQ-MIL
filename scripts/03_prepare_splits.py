#!/usr/bin/env python3
"""
Script 03 — Prepare 5-fold cross-validation splits.

Splits at the patient level (slide_id prefix before the first underscore
is treated as patient ID). Within each fold:
  - 20% of patients → test set
  - Of the remaining 80%: 10% → validation, 90% → train

Output (one directory per fold)::

    splits/
        fold0_train.csv
        fold0_val.csv
        fold0_test.csv
        fold1_train.csv
        ...

Each CSV contains columns: slide_id, label.

Usage::

    python scripts/03_prepare_splits.py \
        --labels data/labels.csv \
        --output_dir data/splits \
        --n_folds 5 \
        --seed 42
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 5-fold CV splits at patient level.")
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"),
                   help="CSV with columns: slide_id, label.")
    p.add_argument("--output_dir", type=Path, default=Path("data/splits"))
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.10,
                   help="Fraction of train+val to use as validation.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def infer_patient_id(slide_id: str) -> str:
    """Extract patient ID from slide_id (prefix before first '_')."""
    parts = str(slide_id).split("_")
    return parts[0] if len(parts) > 1 else slide_id


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.labels)
    df["slide_id"] = df["slide_id"].astype(str)
    df["patient_id"] = df["slide_id"].apply(infer_patient_id)

    # Aggregate to patient level (majority label per patient)
    patient_df = (
        df.groupby("patient_id")["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"label": "patient_label"})
    )

    patients = patient_df["patient_id"].values
    plabels = patient_df["patient_label"].values

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(patients, plabels)):
        test_patients = set(patients[test_idx])
        trainval_patients = patients[trainval_idx]
        trainval_labels = plabels[trainval_idx]

        # Sub-split trainval into train / val (stratified)
        n_val = max(1, int(len(trainval_patients) * args.val_ratio))
        val_skf = StratifiedKFold(n_splits=max(2, round(1 / args.val_ratio)), shuffle=True,
                                  random_state=args.seed + fold)
        # Take the first split as val
        for train_sub_idx, val_sub_idx in val_skf.split(trainval_patients, trainval_labels):
            break  # only first split

        train_patients = set(trainval_patients[train_sub_idx])
        val_patients = set(trainval_patients[val_sub_idx])

        # Assign slides to splits
        train_rows = df[df["patient_id"].isin(train_patients)][["slide_id", "label"]]
        val_rows   = df[df["patient_id"].isin(val_patients)][["slide_id", "label"]]
        test_rows  = df[df["patient_id"].isin(test_patients)][["slide_id", "label"]]

        train_rows.to_csv(args.output_dir / f"fold{fold}_train.csv", index=False)
        val_rows.to_csv(args.output_dir / f"fold{fold}_val.csv", index=False)
        test_rows.to_csv(args.output_dir / f"fold{fold}_test.csv", index=False)

        logger.info(
            "Fold %d — train: %d slides (%d patients), val: %d slides (%d patients), "
            "test: %d slides (%d patients)",
            fold,
            len(train_rows), len(train_patients),
            len(val_rows), len(val_patients),
            len(test_rows), len(test_patients),
        )

    logger.info("Splits saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
