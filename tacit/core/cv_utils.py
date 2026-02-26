"""Computer vision utilities for PNG-based verification.

Provides shared primitives for extracting structural information from
rasterized PNG puzzle images.  Used by all generator verify() methods
after the Track 1 refactoring from SVG-parsing to CV-based parsing.
"""
from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def png_to_numpy(png_bytes: bytes) -> NDArray[np.uint8]:
    """Load PNG bytes into an (H, W, 3) RGB numpy array."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def resize_to(
    img: NDArray[np.uint8], width: int, height: int
) -> NDArray[np.uint8]:
    """Resize image to exact (width, height) dimensions."""
    pil = Image.fromarray(img).resize((width, height), Image.Resampling.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def sample_color(
    img: NDArray[np.uint8], x: int, y: int
) -> tuple[int, int, int]:
    """Sample RGB at pixel (x, y).  img is (H, W, 3) so indexed [y, x]."""
    h, w = img.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' hex string to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def color_distance(
    c1: tuple[int, ...], c2: tuple[int, ...]
) -> float:
    """Euclidean distance between two RGB tuples."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2))))


def find_closest_palette_color(
    pixel: tuple[int, int, int],
    palette: dict[str, int],
    threshold: float = 60.0,
) -> int | None:
    """Map pixel RGB to nearest palette entry index.

    Args:
        pixel: (R, G, B) tuple.
        palette: hex color string -> integer index.
        threshold: max Euclidean distance to accept.

    Returns:
        Palette index, or None if no match within threshold.
    """
    best_idx: int | None = None
    best_dist = float("inf")
    for hex_color, idx in palette.items():
        rgb = hex_to_rgb(hex_color)
        d = color_distance(pixel, rgb)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx if best_dist <= threshold else None


def count_color_pixels(
    img: NDArray[np.uint8],
    target_rgb: tuple[int, int, int],
    threshold: float = 50.0,
) -> int:
    """Count pixels within Euclidean threshold of a target color."""
    diff = img.astype(np.float64) - np.array(target_rgb, dtype=np.float64)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return int((dist < threshold).sum())


def compute_ssim(png1: bytes, png2: bytes) -> float:
    """Structural similarity index between two PNG images.

    Resizes both to matching dimensions if they differ.
    """
    from skimage.metrics import structural_similarity

    img1 = png_to_numpy(png1)
    img2 = png_to_numpy(png2)
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = resize_to(img1, w, h)
        img2 = resize_to(img2, w, h)
    return float(structural_similarity(img1, img2, channel_axis=2))
