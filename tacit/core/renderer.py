"""Thin SVG/PNG rendering abstraction for TACIT Benchmark.

All generators use this module for visual output. Ensures consistent
style (colors, line weights, fonts) across all 10 puzzle types.
Wraps svgwrite for SVG generation and cairosvg for PNG rasterization.
"""
from __future__ import annotations

from pathlib import Path

import svgwrite
from svgwrite import Drawing
import cairosvg

# --- Shared visual style ---
STYLE: dict = {
    "background": "#FFFFFF",
    "line_width": 2,
    "line_color": "#222222",
    "grid_color": "#CCCCCC",
    "highlight_color": "#FF4444",
    "solution_color": "#2266FF",
    "font_family": "monospace",
    "font_size": 14,
    "colors": [
        "#E63946",  # red
        "#457B9D",  # steel blue
        "#2A9D8F",  # teal
        "#E9C46A",  # yellow
        "#F4A261",  # orange
        "#264653",  # dark teal
        "#6A0572",  # purple
        "#1B998B",  # mint
        "#FF6B6B",  # coral
        "#4ECDC4",  # turquoise
    ],
}


def create_canvas(width: int, height: int) -> Drawing:
    """Create an SVG canvas with consistent background."""
    dwg = svgwrite.Drawing(size=(f"{width}", f"{height}"))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=STYLE["background"]))
    return dwg


def draw_rect(
    canvas: Drawing,
    x: float, y: float,
    width: float, height: float,
    fill: str = "none",
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw a rectangle on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.rect(
        insert=(x, y),
        size=(width, height),
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_circle(
    canvas: Drawing,
    cx: float, cy: float, r: float,
    fill: str = "none",
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw a circle on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.circle(
        center=(cx, cy),
        r=r,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_line(
    canvas: Drawing,
    x1: float, y1: float,
    x2: float, y2: float,
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw a line on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.line(
        start=(x1, y1),
        end=(x2, y2),
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_path(
    canvas: Drawing,
    d: str,
    fill: str = "none",
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw an SVG path on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.path(
        d=d,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_text(
    canvas: Drawing,
    x: float, y: float,
    text: str,
    font_size: float | None = None,
    fill: str | None = None,
    anchor: str = "middle",
) -> None:
    """Draw text on the canvas."""
    font_size = font_size or STYLE["font_size"]
    fill = fill or STYLE["line_color"]
    canvas.add(canvas.text(
        text,
        insert=(x, y),
        font_size=f"{font_size}px",
        font_family=STYLE["font_family"],
        fill=fill,
        text_anchor=anchor,
    ))


def svg_to_string(canvas: Drawing) -> str:
    """Convert SVG canvas to string."""
    return canvas.tostring()


def svg_to_png(canvas: Drawing, width: int) -> bytes:
    """Rasterize SVG canvas to PNG bytes at the given width."""
    svg_bytes = canvas.tostring().encode("utf-8")
    return cairosvg.svg2png(bytestring=svg_bytes, output_width=width)


def svg_to_png_multi(canvas: Drawing, widths: list[int]) -> dict[int, bytes]:
    """Rasterize SVG canvas to multiple PNG resolutions."""
    svg_bytes = canvas.tostring().encode("utf-8")
    return {
        w: cairosvg.svg2png(bytestring=svg_bytes, output_width=w)
        for w in widths
    }


def save_svg(canvas: Drawing, path: Path) -> None:
    """Save SVG canvas to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canvas.tostring())


def save_png(canvas: Drawing, path: Path, width: int) -> None:
    """Save SVG canvas as rasterized PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(svg_to_png(canvas, width))


def svg_string_to_png(svg_string: str, width: int | None = None) -> bytes:
    """Rasterize an SVG string to PNG bytes.

    If *width* is None, renders at the SVG's natural viewport size
    (preserving 1:1 pixel mapping with rendering coordinates).
    """
    kwargs: dict = {"bytestring": svg_string.encode("utf-8")}
    if width is not None:
        kwargs["output_width"] = width
    return cairosvg.svg2png(**kwargs)
