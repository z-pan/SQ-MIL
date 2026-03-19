#!/usr/bin/env python3
"""
Script 07 — Generate per-subtype attention heatmaps for test WSIs.

For each WSI in --wsi_dir, looks up the corresponding _inst_pred.csv
in --result_dir and produces a colored overlay PNG.

Usage::

    python scripts/07_generate_heatmaps.py \
        --wsi_dir 'data/wsi/*.tif' \
        --result_dir results/stage2/fold0 \
        --output_dir results/heatmaps \
        --labels data/labels.csv \
        [--gt_dir data/spatial_annotations]  # optional ground-truth

Output per WSI:
  {output_dir}/{slide_id}_heatmap.png   — thumbnail + colored overlay [+ GT side-by-side]
  Already saved in result_dir:
  {result_dir}/{slide_id}_inst_pred.csv — per-patch (x, y, predicted_class, prob_*)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate subtype heatmaps for .tif WSIs.")
    p.add_argument("--wsi_dir", type=str, required=True,
                   help="Glob pattern for .tif WSIs, e.g. 'data/wsi/*.tif'")
    p.add_argument("--result_dir", type=Path, required=True,
                   help="Directory containing {slide_id}_inst_pred.csv files.")
    p.add_argument("--output_dir", type=Path, default=Path("results/heatmaps"))
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--gt_dir", type=Path, default=None,
                   help="Optional: directory with ground-truth annotation CSVs.")
    p.add_argument("--thumbnail_max", type=int, default=1024)
    p.add_argument("--overlay_alpha", type=float, default=0.45)
    p.add_argument("--gaussian_radius", type=float, default=5.0)
    p.add_argument("--patch_size", type=int, default=512)
    return p.parse_args()


def main() -> None:
    import glob
    import pandas as pd
    from tqdm import tqdm
    from src.visualization.heatmap import generate_heatmap

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wsi_paths = sorted(Path(p) for p in glob.glob(args.wsi_dir))
    if not wsi_paths:
        logger.error("No .tif files matched: %s", args.wsi_dir)
        sys.exit(1)
    logger.info("Found %d WSI files.", len(wsi_paths))

    labels_df = pd.read_csv(args.labels) if args.labels.exists() else None

    failed = []
    for wsi_path in tqdm(wsi_paths, desc="Heatmaps"):
        slide_id = wsi_path.stem
        inst_csv = args.result_dir / f"{slide_id}_inst_pred.csv"

        if not inst_csv.exists():
            logger.warning("No prediction CSV for %s — skipping.", slide_id)
            continue

        gt_csv = None
        if args.gt_dir is not None:
            candidate = args.gt_dir / f"{slide_id}_gt.csv"
            if candidate.exists():
                gt_csv = candidate

        out_path = args.output_dir / f"{slide_id}_heatmap.png"

        try:
            generate_heatmap(
                wsi_path=wsi_path,
                inst_csv=inst_csv,
                output_path=out_path,
                gt_csv=gt_csv,
                thumbnail_max=args.thumbnail_max,
                overlay_alpha=args.overlay_alpha,
                gaussian_radius=args.gaussian_radius,
                patch_size=args.patch_size,
            )
        except Exception as exc:
            logger.error("Failed to generate heatmap for %s: %s", slide_id, exc)
            failed.append(slide_id)

    logger.info(
        "Done. %d heatmaps generated, %d failed.",
        len(wsi_paths) - len(failed),
        len(failed),
    )
    if failed:
        logger.warning("Failed slides: %s", failed)


if __name__ == "__main__":
    main()
