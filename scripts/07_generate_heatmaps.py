#!/usr/bin/env python3
"""
Script 07 — Generate per-subtype attention heatmaps for test WSIs.

For every ``.tif`` WSI matched by ``--wsi_dir``, this script:

1. Locates the matching instance-prediction CSV(s) under ``--predictions_dir``.
2. Generates a PNG figure containing:
   - Left panel  : original WSI thumbnail.
   - Right panel : spatial prediction overlay (argmax class color, alpha ∝ confidence).
   - (Optional 3rd panel): ground-truth overlay when ``--gt_dir`` is supplied.
   - Bottom legend : per-class color swatch + mean confidence.
   - Title : slide_id + WSI-level predicted subtype + confidence %.
3. Saves the PNG to ``--output_dir/{slide_id}_heatmap.png``.

Prediction CSV formats accepted
--------------------------------
(a) Trainer output (``SMMILeTrainer.evaluate``):
    ``slide_id, x, y, predicted_class, prob_CC, prob_EC, prob_HGSC, prob_LGSC, prob_MC``

(b) Alternative / upstream format:
    ``filename, x, y, patch_size, pred_class, prob_CC, prob_EC, prob_HGSC, prob_LGSC, prob_MC``

Both formats are auto-detected by :class:`~src.visualization.heatmap.HeatmapGenerator`.

Usage
-----
All 5 folds (trainer inst_predictions CSVs live in results/stage2/)::

    python scripts/07_generate_heatmaps.py \\
        --wsi_dir 'data/wsi/*.tif' \\
        --predictions_dir results/stage2/ \\
        --output_dir results/heatmaps/

Single fold with explicit CSV directory + ground truth::

    python scripts/07_generate_heatmaps.py \\
        --wsi_dir 'data/wsi/*.tif' \\
        --predictions_dir results/stage2/fold0 \\
        --output_dir results/heatmaps/fold0 \\
        --gt_dir data/spatial_annotations \\
        --thumbnail_size 2048 \\
        --num_workers 8

Generate for a single slide::

    python scripts/07_generate_heatmaps.py \\
        --wsi_dir data/wsi/slide_001.tif \\
        --predictions_dir results/stage2/fold0/inst_predictions_fold0.csv \\
        --output_dir results/heatmaps/ \\
        --slide_id slide_001
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-subtype spatial prediction heatmaps for .tif WSIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--wsi_dir", required=True,
        help=(
            "Glob pattern for .tif WSIs (e.g. 'data/wsi/*.tif'), a single .tif "
            "file path, or a directory containing .tif files."
        ),
    )
    p.add_argument(
        "--predictions_dir", required=True,
        help=(
            "Directory containing instance-prediction CSVs produced by "
            "SMMILeTrainer.evaluate (inst_predictions_fold*.csv), "
            "or a single CSV file path."
        ),
    )

    # Optional
    p.add_argument(
        "--output_dir", default="results/heatmaps",
        help="Directory for output .png heatmap files.",
    )
    p.add_argument(
        "--gt_dir", default=None,
        help=(
            "Optional: directory with ground-truth annotation CSVs. "
            "When a matching file is found for a slide, a third GT panel "
            "is added to the figure."
        ),
    )
    p.add_argument(
        "--slide_id", default=None,
        help="Process only this one slide (stem of the .tif filename).",
    )
    p.add_argument(
        "--thumbnail_size", type=int, default=2048,
        help="Max width/height of the WSI thumbnail in pixels (aspect ratio preserved).",
    )
    p.add_argument(
        "--overlay_alpha", type=float, default=0.50,
        help="Maximum opacity of the color overlay [0=transparent, 1=opaque].",
    )
    p.add_argument(
        "--gaussian_sigma", type=float, default=5.0,
        help="Gaussian smoothing sigma in thumbnail pixels.",
    )
    p.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel worker threads.",
    )
    p.add_argument(
        "--labels", default="data/labels.csv",
        help="Optional labels.csv for annotating true-class in the title.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_wsi_paths(wsi_dir: str, slide_id_filter: str | None) -> list[Path]:
    """Resolve WSI paths from a glob pattern, file, or directory."""
    if "*" in wsi_dir or "?" in wsi_dir:
        paths = sorted(Path(p) for p in glob.glob(wsi_dir))
    else:
        p = Path(wsi_dir)
        if p.is_dir():
            paths = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
        elif p.is_file():
            paths = [p]
        else:
            logger.error("wsi_dir not found: %s", wsi_dir)
            sys.exit(1)

    if slide_id_filter:
        paths = [p for p in paths if p.stem == slide_id_filter]

    return paths


def _load_labels(labels_path: str) -> dict[str, str]:
    """Return {slide_id: label_str} from labels.csv, or empty dict."""
    p = Path(labels_path)
    if not p.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(p)
        return {str(row["slide_id"]): str(row["label"]) for _, row in df.iterrows()}
    except Exception as exc:
        logger.warning("Could not load labels.csv: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Insert project root into sys.path so src.* imports work when the script
    # is run directly (python scripts/07_generate_heatmaps.py).
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.visualization.heatmap import HeatmapGenerator

    args = parse_args()

    # ---- Resolve WSI paths -----------------------------------------------
    wsi_paths = _resolve_wsi_paths(args.wsi_dir, args.slide_id)
    if not wsi_paths:
        logger.error(
            "No .tif files matched: %s%s",
            args.wsi_dir,
            f" (slide_id filter: {args.slide_id})" if args.slide_id else "",
        )
        sys.exit(1)
    logger.info("Found %d WSI file(s).", len(wsi_paths))

    # ---- Handle single-CSV prediction file --------------------------------
    pred_input = Path(args.predictions_dir)
    if pred_input.is_file() and pred_input.suffix == ".csv":
        # User passed a single CSV; wrap it in a temporary directory context
        # by setting predictions_dir to its parent and filtering later.
        predictions_dir = pred_input.parent
        # Pre-load the CSV to pass directly to generate() for single-slide
        # mode (avoids rescanning).
        import pandas as pd
        single_pred_df = pd.read_csv(pred_input)
        predictions_dir_str = str(predictions_dir)
    else:
        single_pred_df = None
        predictions_dir_str = str(args.predictions_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dir_p = Path(args.gt_dir) if args.gt_dir else None

    # ---- Create generator ------------------------------------------------
    gen = HeatmapGenerator(
        thumbnail_size=(args.thumbnail_size, args.thumbnail_size),
        overlay_alpha=args.gaussian_sigma if False else args.overlay_alpha,  # noqa
        gaussian_sigma=args.gaussian_sigma,
    )

    # ---- Labels (for informational logging) ------------------------------
    label_map = _load_labels(args.labels)

    # ---- Single-slide shortcut ------------------------------------------
    if len(wsi_paths) == 1 and single_pred_df is not None:
        wsi_path = wsi_paths[0]
        slide_id = wsi_path.stem
        true_label = label_map.get(slide_id, "?")
        logger.info("Processing slide %s (true label: %s)", slide_id, true_label)

        # Filter to this slide if the CSV has slide_id column
        import pandas as pd
        if "slide_id" in single_pred_df.columns:
            slide_df = single_pred_df[single_pred_df["slide_id"] == slide_id]
            if slide_df.empty:
                # Try all rows (user may have passed a single-slide CSV)
                slide_df = single_pred_df
        else:
            slide_df = single_pred_df

        gt_df = None
        if gt_dir_p is not None:
            gt_csv = gt_dir_p / f"{slide_id}_gt.csv"
            if gt_csv.exists():
                gt_df = pd.read_csv(gt_csv)

        out_path = output_dir / f"{slide_id}_heatmap.png"
        try:
            gen.generate(
                wsi_path       = wsi_path,
                predictions_df = slide_df.reset_index(drop=True),
                output_path    = out_path,
                show_gt        = gt_df is not None,
                gt_df          = gt_df,
            )
            logger.info("Done: %s", out_path)
        except Exception as exc:
            logger.error("Failed for %s: %s", slide_id, exc)
            sys.exit(1)
        return

    # ---- Batch mode ------------------------------------------------------
    results = gen.generate_batch(
        wsi_dir         = args.wsi_dir,
        predictions_dir = predictions_dir_str,
        output_dir      = str(output_dir),
        num_workers     = args.num_workers,
        gt_dir          = str(gt_dir_p) if gt_dir_p else None,
    )

    # ---- Summary ---------------------------------------------------------
    ok_count  = sum(1 for v in results.values() if not isinstance(v, Exception))
    err_count = len(results) - ok_count

    logger.info("=" * 50)
    logger.info("Heatmap generation complete.")
    logger.info("  Generated : %d", ok_count)
    logger.info("  Failed    : %d", err_count)
    logger.info("  Skipped   : %d (no predictions)", len(wsi_paths) - len(results))
    logger.info("  Output dir: %s", output_dir)

    if err_count:
        logger.warning(
            "Failed slides: %s",
            [sid for sid, v in results.items() if isinstance(v, Exception)],
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
