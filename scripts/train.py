#!/usr/bin/env python3
"""
Unified SMMILe training / evaluation entry point.

Invoked by shell scripts 04–06, but can also be run directly.

Usage examples
--------------
Stage 1 (fold 0):
    python scripts/train.py \\
        --config configs/ovarian_conch_s1.yaml \\
        --stage 1 --fold 0

Stage 2 (fold 0), loading Stage 1 checkpoint:
    python scripts/train.py \\
        --config configs/ovarian_conch_s2.yaml \\
        --stage 2 --fold 0 \\
        --stage1_ckpt results/stage1/fold0/best_model.pth

Evaluation only (fold 0):
    python scripts/train.py \\
        --config configs/ovarian_conch_s2.yaml \\
        --stage eval --fold 0 \\
        --ckpt results/stage2/fold0/best_model.pth

Config overrides (any of these can be supplied independently):
    --data_root /path/to/data
    --output_dir results/my_run
    --gpu_id 1
    --epochs 60
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ---- resolve project root so this script can be run from any working dir ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import SMMILeTrainer  # noqa: E402
from src.datasets.mil_dataset import build_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, fold_idx: int, stage: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"train_s{stage}_fold{fold_idx}.log"
    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides onto the loaded YAML config in-place."""
    paths = cfg.setdefault("paths", {})
    hw    = cfg.setdefault("hardware", {})
    train = cfg.setdefault("training", {})

    if args.data_root:
        paths["data_root"]       = args.data_root
        paths["embedding_dir"]   = str(Path(args.data_root) / "embeddings")
        paths["superpixel_dir"]  = str(Path(args.data_root) / "superpixels")
        paths["split_dir"]       = str(Path(args.data_root) / "splits")
        paths["labels_csv"]      = str(Path(args.data_root) / "labels.csv")
    # Fine-grained overrides (take precedence over data_root-derived paths)
    if args.emb_dir:
        paths["embedding_dir"]  = args.emb_dir
    if args.sp_dir:
        paths["superpixel_dir"] = args.sp_dir
    if args.output_dir:
        paths["output_dir"] = args.output_dir
    if args.gpu_id is not None:
        hw["gpu_id"] = args.gpu_id
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.fold is not None:
        cfg.setdefault("dataset", {})["fold"] = args.fold
    return cfg


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loader(
    cfg: dict,
    fold_idx: int,
    split: str,
    weighted_sampling: bool = False,
) -> DataLoader:
    """Build a :class:`DataLoader` for *split* of fold *fold_idx*.

    Parameters
    ----------
    cfg:
        Full config dict.
    fold_idx:
        Fold index (0-based).
    split:
        ``'train'``, ``'val'``, or ``'test'``.
    weighted_sampling:
        When True and split=='train', attach a
        :class:`WeightedRandomSampler` for class-imbalance correction.
    """
    paths = cfg["paths"]
    hw    = cfg.get("hardware", {})

    split_csv   = Path(paths["split_dir"]) / f"splits_{fold_idx}.csv"
    labels_csv  = paths["labels_csv"]
    emb_dir     = paths["embedding_dir"]
    sp_dir      = paths["superpixel_dir"]

    augment = split == "train"
    dataset = build_dataset(
        split_csv      = split_csv,
        labels_csv     = labels_csv,
        embedding_dir  = emb_dir,
        superpixel_dir = sp_dir,
        split          = split,
        augment        = augment,
    )

    sampler = None
    shuffle = False
    if weighted_sampling and split == "train":
        sampler = SMMILeTrainer.build_weighted_sampler(dataset)
        # WeightedRandomSampler sets its own iteration order — do not shuffle
    elif split == "train":
        shuffle = True

    return DataLoader(
        dataset,
        batch_size  = 1,           # MIL: one WSI per step
        sampler     = sampler,
        shuffle     = shuffle,
        num_workers = hw.get("num_workers", 4),
        pin_memory  = hw.get("pin_memory", True) and torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Summary: print cross-fold mean ± std after all folds complete
# ---------------------------------------------------------------------------

def print_fold_summary(results: list[dict]) -> None:
    from src.training.evaluator import EvalResult, summarize_folds

    fold_results = [
        EvalResult(
            wsi_auc         = r["wsi_auc"],
            patch_auc       = r["patch_auc"],
            patch_f1        = r["patch_f1"],
            patch_acc       = r["patch_acc"],
            patch_precision = r["patch_precision"],
            patch_recall    = r["patch_recall"],
        )
        for r in results
    ]
    summary = summarize_folds(fold_results)
    print("\n========= Cross-fold summary =========")
    for metric, val in summary.items():
        print(f"  {metric:20s}: {val}")
    print("======================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SMMILe training and evaluation entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      required=True,
                   help="Path to YAML config file.")
    p.add_argument("--stage",       required=True,
                   choices=["1", "2", "eval"],
                   help="Training stage or evaluation mode.")
    p.add_argument("--fold",        type=int, default=None,
                   help="Fold index to run (0-based). Defaults to config value.")
    p.add_argument("--all_folds",   action="store_true",
                   help="Run all 5 folds sequentially and print summary.")

    # Stage 2 / eval checkpoint paths
    p.add_argument("--stage1_ckpt", default=None,
                   help="Path to Stage 1 best checkpoint (required for --stage 2).")
    p.add_argument("--ckpt",        default=None,
                   help="Path to checkpoint to load for evaluation.")

    # Config overrides
    p.add_argument("--data_root",   default=None,
                   help="Override paths.data_root (all sub-paths inferred).")
    p.add_argument("--emb_dir",     default=None,
                   help="Override paths.embedding_dir directly "
                        "(use when embeddings are not under data_root/embeddings).")
    p.add_argument("--sp_dir",      default=None,
                   help="Override paths.superpixel_dir directly "
                        "(use when superpixels are not under data_root/superpixels).")
    p.add_argument("--output_dir",  default=None,
                   help="Override paths.output_dir.")
    p.add_argument("--gpu_id",      type=int, default=None,
                   help="GPU device index (-1 = CPU).")
    p.add_argument("--epochs",      type=int, default=None,
                   help="Override training.epochs.")
    p.add_argument("--seed",        type=int, default=None,
                   help="Override random seed (default: from config experiment.seed).")

    return p.parse_args()


def run_fold(cfg: dict, fold_idx: int, args: argparse.Namespace) -> dict | None:
    """Run one fold — train stage 1/2 or evaluate."""
    stage = args.stage
    paths = cfg["paths"]

    # Output dir for this fold
    out_dir = Path(paths["output_dir"]) / f"fold{fold_idx}"
    setup_logging(out_dir, fold_idx, stage)
    logger = logging.getLogger(__name__)
    logger.info("Config: %s | Stage: %s | Fold: %d", args.config, stage, fold_idx)

    # Reproducibility
    seed = args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed + fold_idx)  # different seed per fold

    use_weighted = cfg["training"].get("weighted_sampling", True)
    trainer = SMMILeTrainer(model=None, config=cfg, fold_idx=fold_idx)

    if stage == "1":
        train_loader = make_loader(cfg, fold_idx, "train", weighted_sampling=use_weighted)
        val_loader   = make_loader(cfg, fold_idx, "val")
        trainer.train_stage1(train_loader, val_loader)
        return None   # evaluation is done separately after Stage 2

    elif stage == "2":
        if args.stage1_ckpt is None:
            # Auto-infer from paths
            s1_base = paths.get(
                "stage1_ckpt",
                f"results/stage1/fold{fold_idx}/best_model.pth",
            )
            s1_path = str(s1_base).replace("{fold}", str(fold_idx))
        else:
            s1_path = args.stage1_ckpt.replace("{fold}", str(fold_idx))

        train_loader = make_loader(cfg, fold_idx, "train", weighted_sampling=use_weighted)
        val_loader   = make_loader(cfg, fold_idx, "val")
        trainer.train_stage2(train_loader, val_loader, s1_path)

        # Run evaluation on test set immediately after Stage 2
        test_loader  = make_loader(cfg, fold_idx, "test")
        metrics      = trainer.evaluate(test_loader)
        logger.info("Fold %d metrics: %s", fold_idx, metrics)
        return metrics

    elif stage == "eval":
        if args.ckpt is None:
            # Auto-infer Stage 2 best checkpoint
            ckpt_path = out_dir / "best_model.pth"
        else:
            ckpt_path = Path(args.ckpt.replace("{fold}", str(fold_idx)))

        ckpt_stage = 2 if (ckpt_path.parent / "best_model.pth").exists() else 1
        # Detect stage from checkpoint metadata
        try:
            ck = torch.load(ckpt_path, map_location="cpu")
            ckpt_stage = ck.get("stage", 2)
        except Exception:
            pass

        trainer.load_checkpoint(ckpt_path, stage=ckpt_stage)
        test_loader = make_loader(cfg, fold_idx, "test")
        metrics     = trainer.evaluate(test_loader)
        logger.info("Fold %d metrics: %s", fold_idx, metrics)
        return metrics

    return None


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args)

    n_folds = cfg.get("dataset", {}).get("n_folds", 5)

    if args.all_folds:
        folds = list(range(n_folds))
    elif args.fold is not None:
        folds = [args.fold]
    else:
        folds = [cfg.get("dataset", {}).get("fold", 0)]

    all_metrics: list[dict] = []
    for fold in folds:
        result = run_fold(cfg, fold, args)
        if result is not None:
            all_metrics.append(result)

    if len(all_metrics) > 1:
        print_fold_summary(all_metrics)

        # Save aggregated results
        out_base = Path(cfg["paths"]["output_dir"])
        agg_path = out_base / "all_folds_metrics.json"
        with open(agg_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Aggregated metrics saved → {agg_path}")


if __name__ == "__main__":
    main()
