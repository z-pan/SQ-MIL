#!/usr/bin/env python3
"""
Script 01 — Extract patch embeddings from .tif WSIs.

Tessellates each WSI into non-overlapping (or strided) patches, filters
background via Otsu thresholding, and encodes tissue patches with a
pretrained encoder, saving each embedding as a .npy file.

Supported encoders
------------------
resnet50  — ImageNet pretrained, third residual block (layer3) + global
            average pool → 1024-dim.  Always available.
conch     — Conch pathology foundation model (ViT-B/16), 512-dim.
            Requires HuggingFace access to MahmoodLab/conch.
            See --help for instructions if the model cannot be loaded.

Output layout
-------------
<output_dir>/
    <slide_id>/
        coords.csv           # columns: x, y, patch_size
        <x>_<y>_<size>.npy  # float32 embedding vector per tissue patch

Usage
-----
    python scripts/01_extract_features.py \\
        --encoder_name resnet50 \\
        --wsi_dir      data/wsi \\
        --output_dir   data/embeddings \\
        --patch_size   512 \\
        --step_size    512 \\
        --level        0 \\
        --batch_size   64 \\
        --device       cuda:0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conch setup instructions (shown when the model is unavailable)
# ---------------------------------------------------------------------------
_CONCH_HELP = """
╔══════════════════════════════════════════════════════════════════════════╗
║  Conch model unavailable — here is how to obtain access:                ║
║                                                                          ║
║  1.  Request access at https://huggingface.co/MahmoodLab/conch           ║
║      (institutional e-mail required; approval is usually < 24 h).       ║
║                                                                          ║
║  2.  Install the Conch package and its dependencies:                     ║
║          pip install conch timm huggingface_hub                          ║
║                                                                          ║
║  3.  Authenticate with HuggingFace:                                      ║
║          huggingface-cli login                                           ║
║                                                                          ║
║  4.  Re-run this script with --encoder_name conch.                       ║
║                                                                          ║
║  In the meantime, use --encoder_name resnet50 (1024-dim, always works).  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract patch embeddings from .tif WSIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--encoder_name", choices=["resnet50", "conch"], default="resnet50",
        help="Encoder backbone.  resnet50 is always available; "
             "conch requires HuggingFace access (run --help for details).",
    )
    p.add_argument(
        "--wsi_dir", type=Path, required=True,
        help="Directory containing *.tif WSI files.",
    )
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Root output directory for embeddings.",
    )
    p.add_argument("--patch_size", type=int, default=512,
                   help="Patch edge length in level-0 pixels.")
    p.add_argument("--step_size",  type=int, default=512,
                   help="Grid stride in level-0 pixels (== patch_size for no overlap).")
    p.add_argument("--level",      type=int, default=0,
                   help="Pyramid level passed to WSIReader.read_region.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Number of patches per encoder forward pass.")
    p.add_argument("--device",     type=str, default="cuda:0",
                   help="PyTorch device string (cuda:0 / cpu).")
    p.add_argument("--tissue_threshold", type=float, default=0.7,
                   help="Minimum foreground-pixel fraction to keep a patch.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def _load_resnet50(device: str):
    """ResNet-50 truncated at layer3 + global average pool → 1024-dim."""
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T

    backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    # children order: conv1 bn1 relu maxpool layer1 layer2 layer3 layer4 avgpool fc
    #                 0     1   2    3        4      5      6      7      8       9
    # We want layer3 output (1024 channels) + global avg pool.
    layer3_block = nn.Sequential(*list(backbone.children())[:7])  # up to layer3
    pool         = nn.AdaptiveAvgPool2d((1, 1))
    flatten      = nn.Flatten(1)
    model        = nn.Sequential(layer3_block, pool, flatten).eval().to(device)

    transform = T.Compose([
        T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform, 1024


def _load_conch(device: str):
    """Conch ViT-B/16 pathology foundation model → 512-dim."""
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        print(_CONCH_HELP)
        raise SystemExit(
            "conch package not installed.  "
            "pip install conch  (see instructions above)."
        ) from exc

    try:
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
        )
    except Exception as exc:
        print(_CONCH_HELP)
        raise SystemExit(
            f"Failed to load Conch from HuggingFace: {exc}\n"
            "See instructions above."
        ) from exc

    model = model.eval().to(device)
    return model, preprocess, 512


def load_encoder(encoder_name: str, device: str):
    """Return (model, transform, embedding_dim) for the requested encoder.

    The model is frozen (no gradients) and moved to *device*.
    """
    import torch

    if encoder_name == "resnet50":
        model, transform, dim = _load_resnet50(device)
    elif encoder_name == "conch":
        model, transform, dim = _load_conch(device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name!r}")

    for p in model.parameters():
        p.requires_grad_(False)

    logger.info("Encoder: %s  |  embedding dim: %d  |  device: %s", encoder_name, dim, device)
    return model, transform, dim


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def _encode_batch(model, patch_tensors, device: str) -> "np.ndarray":
    """Run one batch through the encoder and return float32 numpy array."""
    import torch
    import numpy as np

    batch = torch.stack(patch_tensors).to(device)
    with torch.no_grad():
        if hasattr(model, "encode_image"):
            # Conch CLIP-style interface
            feats = model.encode_image(batch)
        else:
            feats = model(batch)
            if feats.dim() > 2:
                # Safety: apply global avg pool if the head is missing
                feats = feats.mean(dim=list(range(2, feats.dim())))
    return feats.cpu().float().numpy()


def process_slide(
    wsi_path: Path,
    output_dir: Path,
    model,
    transform,
    patch_size: int,
    step_size: int,
    level: int,
    batch_size: int,
    tissue_threshold: float,
    device: str,
) -> int:
    """Extract and save embeddings for all tissue patches in *wsi_path*.

    Returns the number of patches saved (0 if already done).
    Skips the slide if coords.csv already exists (resume-safe).
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # Import here so the script can be imported without src on PYTHONPATH
    # if only load_encoder / parse_args are needed.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.datasets.wsi_utils import WSIReader, is_tissue

    slide_id = wsi_path.stem
    slide_out = output_dir / slide_id
    coords_csv = slide_out / "coords.csv"

    if coords_csv.exists():
        n = len(pd.read_csv(coords_csv))
        logger.info("SKIP  %s — coords.csv exists (%d patches).", slide_id, n)
        return n

    slide_out.mkdir(parents=True, exist_ok=True)

    saved_rows: list[dict] = []
    # Buffer: list of (x, y, tensor)
    pending: list[tuple[int, int]] = []
    pending_tensors: list = []

    def flush() -> None:
        if not pending_tensors:
            return
        feats = _encode_batch(model, pending_tensors, device)
        for i, (x, y) in enumerate(pending):
            np.save(str(slide_out / f"{x}_{y}_{patch_size}.npy"), feats[i])
            saved_rows.append({"x": x, "y": y, "patch_size": patch_size})
        pending.clear()
        pending_tensors.clear()

    with WSIReader(wsi_path) as reader:
        wsi_w, wsi_h = reader.get_dimensions()
        logger.info(
            "Processing  %s  (%d×%d px, backend=%s)",
            slide_id, wsi_w, wsi_h, reader.backend,
        )

        xs = list(range(0, wsi_w - patch_size + 1, step_size))
        ys = list(range(0, wsi_h - patch_size + 1, step_size))
        total_candidates = len(xs) * len(ys)

        with tqdm(total=total_candidates, desc=slide_id, unit="patch", leave=False) as pbar:
            for y in ys:
                for x in xs:
                    patch_img = reader.read_region(
                        location=(x, y),
                        level=level,
                        size=(patch_size, patch_size),
                    )  # returns PIL.Image (RGB)
                    pbar.update(1)

                    if not is_tissue(patch_img, threshold=tissue_threshold):
                        continue

                    pending.append((x, y))
                    pending_tensors.append(transform(patch_img))

                    if len(pending_tensors) >= batch_size:
                        flush()

        flush()  # final partial batch

    coords_df = pd.DataFrame(saved_rows)
    coords_df.to_csv(coords_csv, index=False)
    logger.info(
        "DONE  %s — %d / %d patches kept (tissue threshold=%.2f).",
        slide_id, len(saved_rows), total_candidates, tissue_threshold,
    )
    return len(saved_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import torch

    # Resolve device: fall back to CPU if CUDA is requested but unavailable.
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device = "cpu"

    model, transform, emb_dim = load_encoder(args.encoder_name, device)

    wsi_paths = sorted(args.wsi_dir.glob("*.tif"))
    if not wsi_paths:
        logger.error("No .tif files found in %s", args.wsi_dir)
        sys.exit(1)
    logger.info("Found %d WSI files in %s.", len(wsi_paths), args.wsi_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_patches = 0
    for i, wsi_path in enumerate(wsi_paths, 1):
        logger.info("[%d/%d] %s", i, len(wsi_paths), wsi_path.name)
        try:
            n = process_slide(
                wsi_path=wsi_path,
                output_dir=args.output_dir,
                model=model,
                transform=transform,
                patch_size=args.patch_size,
                step_size=args.step_size,
                level=args.level,
                batch_size=args.batch_size,
                tissue_threshold=args.tissue_threshold,
                device=device,
            )
            total_patches += n
        except Exception:
            logger.exception("Failed to process %s — skipping.", wsi_path.name)

    logger.info("All done.  Total tissue patches extracted: %d", total_patches)


if __name__ == "__main__":
    main()
