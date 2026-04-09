"""
scripts/00_validate_data.py
============================
Validate and (optionally) reorganize pre-extracted HuggingFace embeddings so
they match the directory layout expected by ``src/datasets/mil_dataset.py``.

HuggingFace layout (zeyugao/SMMILe_Datasets)
----------------------------------------------
  <emb_dir>/
      <slide_id>_<patch_idx>_<emb_dim>.npy   # one .npy per patch   (shape: (D,))
        OR
      <slide_id>_0_<emb_dim>.npy             # one .npy per slide   (shape: (N, D))

  <sp_dir>/
      <slide_id>_0_<patch_size>.npy          # one .npy per slide   (shape: (N,))

Expected layout (required by MILDataset)
-----------------------------------------
  <emb_dir>/
      <slide_id>/
          coords.csv        columns: x, y, patch_size
          <x>_<y>_<ps>.npy  one .npy per patch  (shape: (D,))

  <sp_dir>/
      <slide_id>.npy        shape: (N,)

Usage
-----
  # Step 1 — just validate (no writes)
  python scripts/00_validate_data.py \\
      --labels      data/labels.csv \\
      --emb_dir     data/embeddings \\
      --sp_dir      data/superpixels

  # Step 2 — validate + reorganize into expected layout
  python scripts/00_validate_data.py \\
      --labels      data/labels.csv \\
      --emb_dir     data/embeddings \\
      --sp_dir      data/superpixels \\
      --reorganize

  # Use separate output dirs (keeps originals untouched)
  python scripts/00_validate_data.py \\
      --labels      data/labels.csv \\
      --emb_dir     data/hf_embeddings \\
      --sp_dir      data/hf_superpixels \\
      --out_emb_dir data/embeddings \\
      --out_sp_dir  data/superpixels \\
      --reorganize
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_LABELS = {"CC", "EC", "HGSC", "LGSC", "MC"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slide_id_from_stem(stem: str) -> str:
    """Extract slide_id from a flat-layout filename stem.

    Convention: slide_id is everything before the FIRST underscore that
    is followed by a digit.  This handles numeric slide IDs (e.g. '13987')
    as well as alphanumeric ones (e.g. 'TCGA-01-A2B3').

    Examples
    --------
    >>> slide_id_from_stem('13987_0_512')
    '13987'
    >>> slide_id_from_stem('TCGA-01-A2B3_45_512')
    'TCGA-01-A2B3'
    >>> slide_id_from_stem('slide001_0_512')
    'slide001'
    """
    parts = stem.split("_")
    # Walk from left; the slide_id ends just before the first purely-numeric part
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0:
            return "_".join(parts[:i])
    # Fallback: everything before last two segments (index + size)
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return stem


def detect_layout(directory: Path) -> str:
    """Return 'flat', 'structured', or 'empty'.

    'flat'       — .npy files live directly in *directory*
    'structured' — .npy files live in per-slide sub-directories
    'empty'      — no .npy files found at all
    """
    npy_files = list(directory.glob("*.npy"))
    if npy_files:
        return "flat"
    sub_npy = list(directory.rglob("*.npy"))
    if sub_npy:
        return "structured"
    return "empty"


def scan_flat_dir(directory: Path) -> dict[str, list[Path]]:
    """Map slide_id → list of .npy paths for a flat-layout directory."""
    result: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(directory.glob("*.npy")):
        sid = slide_id_from_stem(p.stem)
        result[sid].append(p)
    return dict(result)


def scan_structured_dir(directory: Path) -> dict[str, Path]:
    """Map slide_id → sub-directory path for a structured-layout directory."""
    return {
        d.name: d
        for d in sorted(directory.iterdir())
        if d.is_dir() and any(d.glob("*.npy"))
    }


def probe_npy(path: Path) -> tuple[tuple[int, ...], str]:
    """Return (shape, dtype_str) of a .npy file."""
    try:
        arr = np.load(str(path), mmap_mode="r")
    except ValueError:
        # Object-dtype arrays cannot be memory-mapped; fall back to full load
        arr = np.load(str(path), allow_pickle=True)
    return arr.shape, str(arr.dtype)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    labels_csv: Path,
    emb_dir: Path,
    sp_dir: Path,
) -> tuple[list[str], dict]:
    """Check alignment between labels.csv and the embedding/superpixel dirs.

    Returns
    -------
    aligned_ids : list[str]
        Slide IDs present in ALL three sources (labels, embeddings, superpixels).
    report : dict
        Detailed breakdown for printing.
    """
    # --- 1. Load labels.csv ---
    if not labels_csv.exists():
        logger.error("labels.csv not found: %s", labels_csv)
        sys.exit(1)

    df = pd.read_csv(labels_csv)
    df["slide_id"] = df["slide_id"].astype(str).str.strip()

    missing_cols = {"slide_id", "label"} - set(df.columns)
    if missing_cols:
        logger.error("labels.csv is missing columns: %s", missing_cols)
        sys.exit(1)

    bad_labels = df[~df["label"].isin(VALID_LABELS)]
    if not bad_labels.empty:
        logger.warning(
            "labels.csv has %d rows with unrecognised labels: %s  "
            "(valid: %s)",
            len(bad_labels),
            bad_labels["label"].unique().tolist(),
            sorted(VALID_LABELS),
        )

    label_ids: set[str] = set(df["slide_id"].tolist())
    label_dist = df.groupby("label")["slide_id"].count().to_dict()

    # --- 2. Scan embeddings ---
    emb_layout = detect_layout(emb_dir)
    if emb_layout == "flat":
        emb_map = scan_flat_dir(emb_dir)
        emb_ids = set(emb_map.keys())
        # Probe one file to learn embedding shape
        sample_sid = next(iter(emb_map))
        sample_files = sorted(emb_map[sample_sid])
        probe_shape, probe_dtype = probe_npy(sample_files[0])
        emb_is_matrix = len(probe_shape) == 2   # (N, D) = all patches in one file
        emb_info = (
            f"flat layout  |  {len(emb_ids)} unique slide_ids  |  "
            f"example file: {sample_files[0].name}  |  "
            f"shape={probe_shape} dtype={probe_dtype}"
        )
    elif emb_layout == "structured":
        emb_struct = scan_structured_dir(emb_dir)
        emb_ids = set(emb_struct.keys())
        emb_map = {}          # not used for structured
        emb_is_matrix = False
        sample_sid = next(iter(emb_struct))
        sub = emb_struct[sample_sid]
        has_coords = (sub / "coords.csv").exists()
        emb_info = (
            f"structured layout  |  {len(emb_ids)} slide dirs  |  "
            f"coords.csv present: {has_coords}"
        )
    else:
        logger.error("No .npy files found in emb_dir: %s", emb_dir)
        sys.exit(1)

    # --- 3. Scan superpixels ---
    sp_layout = detect_layout(sp_dir)
    if sp_layout == "flat":
        sp_map = scan_flat_dir(sp_dir)
        sp_ids = set(sp_map.keys())
        sample_sp_sid = next(iter(sp_map))
        sp_shape, sp_dtype = probe_npy(sp_map[sample_sp_sid][0])
        sp_info = (
            f"flat layout  |  {len(sp_ids)} unique slide_ids  |  "
            f"example file: {sp_map[sample_sp_sid][0].name}  |  "
            f"shape={sp_shape} dtype={sp_dtype}"
        )
    elif sp_layout == "structured":
        # Structured superpixel dir: {slide_id}.npy files
        sp_ids = {p.stem for p in sp_dir.glob("*.npy")}
        sp_map = {}
        sample_sp_sid = next(iter(sp_ids))
        sp_shape, sp_dtype = probe_npy(sp_dir / f"{sample_sp_sid}.npy")
        sp_info = (
            f"structured layout  |  {len(sp_ids)} slide .npy files  |  "
            f"shape={sp_shape} dtype={sp_dtype}"
        )
    else:
        logger.warning("No .npy files found in sp_dir: %s — superpixels will be skipped.", sp_dir)
        sp_ids = set()
        sp_map = {}
        sp_info = "EMPTY"

    # --- 4. Alignment checks ---
    in_labels_only      = label_ids - emb_ids - sp_ids
    in_labels_no_emb    = label_ids - emb_ids
    in_labels_no_sp     = label_ids - sp_ids
    in_emb_no_labels    = emb_ids   - label_ids
    in_sp_no_labels     = sp_ids    - label_ids
    aligned_ids         = sorted(label_ids & emb_ids & (sp_ids if sp_ids else label_ids))

    report = {
        "label_count":       len(label_ids),
        "label_distribution": label_dist,
        "emb_count":         len(emb_ids),
        "sp_count":          len(sp_ids),
        "emb_layout":        emb_layout,
        "sp_layout":         sp_layout,
        "emb_info":          emb_info,
        "sp_info":           sp_info,
        "emb_is_matrix":     emb_is_matrix if emb_layout == "flat" else False,
        "in_labels_no_emb":  sorted(in_labels_no_emb),
        "in_labels_no_sp":   sorted(in_labels_no_sp),
        "in_emb_no_labels":  sorted(in_emb_no_labels),
        "in_sp_no_labels":   sorted(in_sp_no_labels),
        "aligned_count":     len(aligned_ids),
        "emb_map":           emb_map,
        "sp_map":            sp_map if sp_layout == "flat" else {},
    }
    return aligned_ids, report


def print_report(report: dict, aligned_ids: list[str]) -> None:
    sep = "─" * 70

    print(f"\n{sep}")
    print("  DATA VALIDATION REPORT")
    print(sep)

    print(f"\n{'labels.csv':}")
    print(f"  Total slides    : {report['label_count']}")
    print(f"  Class distribution:")
    for lbl, cnt in sorted(report["label_distribution"].items()):
        print(f"    {lbl:6s}: {cnt}")

    print(f"\nEmbeddings directory:")
    print(f"  {report['emb_info']}")

    print(f"\nSuperpixels directory:")
    print(f"  {report['sp_info']}")

    print(f"\n{sep}")
    n_no_emb = len(report["in_labels_no_emb"])
    n_no_sp  = len(report["in_labels_no_sp"])
    n_extra_emb = len(report["in_emb_no_labels"])
    n_extra_sp  = len(report["in_sp_no_labels"])

    status = "✓ PASS" if n_no_emb == 0 and n_no_sp == 0 else "✗ ISSUES FOUND"
    print(f"\n  Alignment: {status}")
    print(f"  Slides aligned (in all sources) : {report['aligned_count']}")
    print(f"  In labels, missing embeddings   : {n_no_emb}")
    print(f"  In labels, missing superpixels  : {n_no_sp}")
    print(f"  In embeddings, not in labels    : {n_extra_emb}  (will be ignored)")
    print(f"  In superpixels, not in labels   : {n_extra_sp}  (will be ignored)")

    if n_no_emb > 0:
        shown = report["in_labels_no_emb"][:20]
        print(f"\n  ⚠  Slides in labels with NO embedding ({n_no_emb}):")
        for sid in shown:
            print(f"       {sid}")
        if n_no_emb > 20:
            print(f"       ... and {n_no_emb - 20} more")

    if n_no_sp > 0:
        shown = report["in_labels_no_sp"][:20]
        print(f"\n  ⚠  Slides in labels with NO superpixel ({n_no_sp}):")
        for sid in shown:
            print(f"       {sid}")
        if n_no_sp > 20:
            print(f"       ... and {n_no_sp - 20} more")

    if report["emb_layout"] == "flat":
        print(f"\n  ⚠  Embeddings are in FLAT layout.")
        print(     "     Run with --reorganize to convert to the expected structured layout.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Reorganization
# ---------------------------------------------------------------------------

def reorganize(
    aligned_ids: list[str],
    report: dict,
    emb_dir: Path,
    sp_dir: Path,
    out_emb_dir: Path,
    out_sp_dir: Path,
) -> None:
    """Convert flat HuggingFace layout to structured layout.

    Embeddings
    ----------
    Two sub-cases detected automatically:

    A) One .npy per patch  (file shape = (D,)):
       Reorganise each file into per-slide sub-dirs.
       Generate a synthetic coords.csv with sequential indices as x-coords
       (real (x,y) pixel coords are unavailable in HuggingFace embeddings;
       sequential indices suffice for training — heatmap generation requires
       real coords and won't be accurate without them).

    B) One .npy per slide  (file shape = (N, D)):
       Split the matrix into per-patch .npy files inside a sub-dir.
       Same synthetic coords.csv caveat as above.

    Superpixels
    -----------
    Rename ``{slide_id}_*_*.npy`` → ``{slide_id}.npy`` in out_sp_dir.
    """
    out_emb_dir.mkdir(parents=True, exist_ok=True)
    out_sp_dir.mkdir(parents=True, exist_ok=True)

    emb_layout   = report["emb_layout"]
    sp_layout    = report["sp_layout"]
    emb_map: dict[str, list[Path]] = report["emb_map"]
    sp_map:  dict[str, list[Path]] = report["sp_map"]
    emb_is_matrix: bool            = report["emb_is_matrix"]

    total = len(aligned_ids)
    logger.info("Reorganising %d slides …", total)

    for i, sid in enumerate(aligned_ids, 1):
        if i % 50 == 0 or i == total:
            logger.info("  %d / %d", i, total)

        # ---- Embeddings ----
        slide_out = out_emb_dir / sid

        if emb_layout == "structured":
            # Already structured — nothing to do unless output dir differs
            src_dir = emb_dir / sid
            if src_dir.resolve() != slide_out.resolve():
                if slide_out.exists():
                    shutil.rmtree(slide_out)
                shutil.copytree(src_dir, slide_out)
        else:
            slide_out.mkdir(parents=True, exist_ok=True)
            files = sorted(emb_map.get(sid, []))

            if not files:
                logger.warning("No embedding files for slide '%s' — skipping.", sid)
                continue

            if emb_is_matrix:
                # Case B: single (N, D) matrix per slide
                mat = np.load(str(files[0]))    # (N, D)
                n_patches = mat.shape[0]
                coords_rows = []
                for idx in range(n_patches):
                    x, y, ps = idx, 0, 512      # synthetic coords
                    np.save(str(slide_out / f"{x}_{y}_{ps}.npy"), mat[idx])
                    coords_rows.append({"x": x, "y": y, "patch_size": ps})
            else:
                # Case A: one (D,) file per patch
                coords_rows = []
                for idx, fpath in enumerate(files):
                    x, y, ps = idx, 0, 512      # synthetic coords
                    dst = slide_out / f"{x}_{y}_{ps}.npy"
                    if not dst.exists():
                        shutil.copy2(str(fpath), str(dst))
                    coords_rows.append({"x": x, "y": y, "patch_size": ps})

            coords_csv = slide_out / "coords.csv"
            if not coords_csv.exists():
                pd.DataFrame(coords_rows).to_csv(str(coords_csv), index=False)

        # ---- Superpixels ----
        sp_out = out_sp_dir / f"{sid}.npy"

        if sp_layout == "structured":
            src = sp_dir / f"{sid}.npy"
            if src.resolve() != sp_out.resolve() and not sp_out.exists():
                shutil.copy2(str(src), str(sp_out))
        elif sp_layout == "flat" and sid in sp_map:
            files_sp = sorted(sp_map[sid])
            if not sp_out.exists():
                # Load, squeeze to 1D, save
                arr = np.load(str(files_sp[0]))
                if arr.ndim == 2:
                    arr = arr.squeeze()
                np.save(str(sp_out), arr.astype(np.int64))

    logger.info("Reorganisation complete.")
    logger.info("  Embeddings → %s", out_emb_dir)
    logger.info("  Superpixels → %s", out_sp_dir)

    if report["emb_layout"] == "flat":
        logger.warning(
            "Synthetic coordinates were used (x=patch_index, y=0). "
            "Training is unaffected. Heatmap generation requires real (x,y) "
            "pixel coordinates — extract them from raw WSIs if needed."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate labels.csv / embedding / superpixel alignment "
                    "and optionally reorganise HuggingFace flat layout to "
                    "MILDataset structured layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--labels",      type=Path, default=Path("data/labels.csv"),
                   help="Path to labels.csv  (default: data/labels.csv)")
    p.add_argument("--emb_dir",     type=Path, default=Path("data/embeddings"),
                   help="Embedding directory  (default: data/embeddings)")
    p.add_argument("--sp_dir",      type=Path, default=Path("data/superpixels"),
                   help="Superpixel directory  (default: data/superpixels)")
    p.add_argument("--out_emb_dir", type=Path, default=None,
                   help="Output embedding dir for reorganisation "
                        "(default: same as --emb_dir, i.e. in-place)")
    p.add_argument("--out_sp_dir",  type=Path, default=None,
                   help="Output superpixel dir for reorganisation "
                        "(default: same as --sp_dir, i.e. in-place)")
    p.add_argument("--reorganize",  action="store_true",
                   help="Convert flat layout to structured layout after validation.")
    p.add_argument("--save_filtered_labels", type=Path, default=None,
                   help="If set, write a filtered labels.csv containing only "
                        "the aligned slide IDs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_emb = args.out_emb_dir or args.emb_dir
    out_sp  = args.out_sp_dir  or args.sp_dir

    # --- Validate ---
    aligned_ids, report = validate(args.labels, args.emb_dir, args.sp_dir)
    print_report(report, aligned_ids)

    # --- Save filtered labels if requested ---
    if args.save_filtered_labels:
        df = pd.read_csv(args.labels)
        df["slide_id"] = df["slide_id"].astype(str).str.strip()
        df_filtered = df[df["slide_id"].isin(set(aligned_ids))]
        args.save_filtered_labels.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(str(args.save_filtered_labels), index=False)
        logger.info(
            "Filtered labels.csv saved (%d slides) → %s",
            len(df_filtered), args.save_filtered_labels,
        )

    # --- Reorganise if requested ---
    if args.reorganize:
        if report["emb_layout"] == "structured" and out_emb.resolve() == args.emb_dir.resolve():
            logger.info("Embeddings already in structured layout — nothing to reorganise.")
        else:
            reorganize(aligned_ids, report, args.emb_dir, args.sp_dir, out_emb, out_sp)
    else:
        if report["emb_layout"] == "flat":
            print("Run with --reorganize to convert flat layout to structured layout.")


if __name__ == "__main__":
    main()
