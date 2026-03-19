#!/usr/bin/env python3
"""
Script 01 — Extract patch embeddings from .tif WSIs.

Tessellates each WSI into non-overlapping 512×512 patches at 20×,
filters background via Otsu thresholding, and saves embeddings as .npy.

Supported encoders:
  conch    — Conch pathology foundation model (512-dim)
  resnet50 — ResNet-50 third residual block (1024-dim, ImageNet weights)

Output layout::

    output_dir/
        {slide_id}/
            coords.csv           # x, y, patch_size columns
            {x}_{y}_{size}.npy  # embedding per patch

Usage::

    python scripts/01_extract_features.py \
        --wsi_dir data/wsi \
        --output_dir data/embeddings \
        --encoder conch \
        --patch_size 512 \
        --mag 20
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract patch embeddings from .tif WSIs.")
    p.add_argument("--wsi_dir", type=Path, required=True,
                   help="Directory containing .tif WSI files.")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output directory for embeddings.")
    p.add_argument("--encoder", choices=["conch", "resnet50"], default="conch",
                   help="Encoder backbone.")
    p.add_argument("--patch_size", type=int, default=512,
                   help="Patch size in pixels at target magnification.")
    p.add_argument("--mag", type=float, default=20.0,
                   help="Target magnification (e.g. 20 for 20×).")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Number of patches per encoder forward pass.")
    p.add_argument("--otsu_threshold", type=float, default=0.8,
                   help="Maximum fraction of white pixels to keep a patch.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device string (cuda / cpu).")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def load_encoder(encoder_name: str, device: str):
    """Load the specified encoder and return (model, transform, embedding_dim)."""
    import torch

    if encoder_name == "resnet50":
        import torchvision.models as tvm
        import torchvision.transforms as T

        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        # Third residual block output (layer3) — 1024 channels
        model = torch.nn.Sequential(*list(backbone.children())[:7])
        model.eval()

        transform = T.Compose([
            T.Resize(512),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        embedding_dim = 1024

    elif encoder_name == "conch":
        from huggingface_hub import hf_hub_download
        import timm
        import torchvision.transforms as T

        # Conch v1 — requires HuggingFace access to MahmoodLab/conch
        # Falls back to ViT-Large from timm if unavailable.
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            model, preprocess = create_model_from_pretrained(
                "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
            )
            transform = preprocess
            embedding_dim = 512
        except Exception as exc:
            logger.warning(
                "Could not load Conch from HuggingFace (%s). "
                "Ensure you have access to MahmoodLab/conch and are logged in "
                "via `huggingface-cli login`.", exc
            )
            raise

    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    return model, transform, embedding_dim


def is_background(patch_rgb, threshold: float = 0.8) -> bool:
    """Return True if the patch is predominantly background (white/near-white)."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_frac = (binary == 255).mean()
    return white_frac > threshold


def process_slide(
    wsi_path: Path,
    output_dir: Path,
    model,
    transform,
    patch_size: int,
    batch_size: int,
    otsu_threshold: float,
    device: str,
) -> int:
    """Extract embeddings for all foreground patches in one WSI.

    Returns number of patches saved.
    """
    import io
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm

    from src.datasets.wsi_utils import WSIReader

    slide_id = wsi_path.stem
    slide_out = output_dir / slide_id
    coords_csv = slide_out / "coords.csv"

    if coords_csv.exists():
        existing = pd.read_csv(coords_csv)
        logger.info("Skipping %s — already extracted (%d patches).", slide_id, len(existing))
        return len(existing)

    slide_out.mkdir(parents=True, exist_ok=True)

    with WSIReader(wsi_path) as reader:
        wsi_w, wsi_h = reader.dimensions
        rows = []
        patch_buffer: list[tuple[int, int, torch.Tensor]] = []

        def flush_buffer():
            if not patch_buffer:
                return
            imgs = torch.stack([t for _, _, t in patch_buffer]).to(device)
            with torch.no_grad():
                if hasattr(model, "encode_image"):
                    feats = model.encode_image(imgs)
                else:
                    feats = model(imgs)
                    if feats.dim() > 2:
                        feats = feats.mean(dim=[2, 3])  # global average pool
            feats = feats.cpu().numpy()
            for i, (x, y, _) in enumerate(patch_buffer):
                npy_path = slide_out / f"{x}_{y}_{patch_size}.npy"
                np.save(str(npy_path), feats[i])
                rows.append({"x": x, "y": y, "patch_size": patch_size})
            patch_buffer.clear()

        xs = range(0, wsi_w - patch_size + 1, patch_size)
        ys = range(0, wsi_h - patch_size + 1, patch_size)
        total = len(xs) * len(ys)

        with tqdm(total=total, desc=slide_id, unit="patch") as pbar:
            for y in ys:
                for x in xs:
                    region = reader.read_region(x, y, patch_size, patch_size, level=0)
                    pbar.update(1)

                    if is_background(region, threshold=otsu_threshold):
                        continue

                    pil_img = Image.fromarray(region)
                    tensor = transform(pil_img)
                    patch_buffer.append((x, y, tensor))

                    if len(patch_buffer) >= batch_size:
                        flush_buffer()

        flush_buffer()

    coords_df = pd.DataFrame(rows)
    coords_df.to_csv(coords_csv, index=False)
    logger.info("%s: saved %d patches.", slide_id, len(rows))
    return len(rows)


def main() -> None:
    args = parse_args()
    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    model, transform, emb_dim = load_encoder(args.encoder, device)
    logger.info("Loaded encoder: %s (%d-dim)", args.encoder, emb_dim)

    wsi_paths = sorted(args.wsi_dir.glob("*.tif"))
    if not wsi_paths:
        logger.error("No .tif files found in %s", args.wsi_dir)
        sys.exit(1)
    logger.info("Found %d WSI files.", len(wsi_paths))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_patches = 0
    for wsi_path in wsi_paths:
        n = process_slide(
            wsi_path=wsi_path,
            output_dir=args.output_dir,
            model=model,
            transform=transform,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            otsu_threshold=args.otsu_threshold,
            device=device,
        )
        total_patches += n

    logger.info("Done. Total patches extracted: %d", total_patches)


if __name__ == "__main__":
    main()
