"""Tests for CV utility functions."""
import numpy as np
import pytest

from tacit.core.cv_utils import (
    color_distance,
    compute_ssim,
    count_color_pixels,
    find_closest_palette_color,
    hex_to_rgb,
    png_to_numpy,
    sample_color,
)
from tacit.core.renderer import svg_string_to_png


def _make_red_png() -> bytes:
    """Create a tiny red PNG for testing."""
    from tacit.core.renderer import create_canvas, draw_rect, svg_to_string
    canvas = create_canvas(10, 10)
    draw_rect(canvas, 0, 0, 10, 10, fill="#FF0000", stroke="none", stroke_width=0)
    svg_str = svg_to_string(canvas)
    return svg_string_to_png(svg_str)


def test_png_to_numpy():
    png = _make_red_png()
    img = png_to_numpy(png)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_hex_to_rgb():
    assert hex_to_rgb("#FF0000") == (255, 0, 0)
    assert hex_to_rgb("#00ff00") == (0, 255, 0)
    assert hex_to_rgb("#2266FF") == (34, 102, 255)


def test_color_distance():
    assert color_distance((255, 0, 0), (255, 0, 0)) == 0.0
    assert color_distance((255, 0, 0), (0, 0, 0)) == pytest.approx(255.0, abs=0.1)


def test_find_closest_palette_color():
    palette = {"#FF0000": 0, "#00FF00": 1, "#0000FF": 2}
    assert find_closest_palette_color((250, 5, 5), palette) == 0
    assert find_closest_palette_color((5, 250, 5), palette) == 1
    assert find_closest_palette_color((128, 128, 128), palette, threshold=30) is None


def test_count_color_pixels():
    png = _make_red_png()
    img = png_to_numpy(png)
    red_count = count_color_pixels(img, (255, 0, 0), threshold=60)
    assert red_count > 0
    blue_count = count_color_pixels(img, (0, 0, 255), threshold=30)
    assert blue_count == 0


def test_sample_color():
    png = _make_red_png()
    img = png_to_numpy(png)
    r, g, b = sample_color(img, 5, 5)
    assert r > 200  # Should be reddish


def test_compute_ssim_identical():
    png = _make_red_png()
    ssim = compute_ssim(png, png)
    assert ssim > 0.99
