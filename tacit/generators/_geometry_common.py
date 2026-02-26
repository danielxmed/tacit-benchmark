"""Shared 3D geometry utilities for orthographic projection and isometric reconstruction.

Provides:
- Voxel model generation (random 3D solids with configurable complexity)
- Orthographic projection (voxel grid -> 2D binary silhouette)
- Isometric rendering (voxel grid -> SVG isometric view)
- 2D projection rendering (binary grid -> SVG)

Used by OrthoProjectionGenerator and IsoReconstructionGenerator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tacit.core import renderer


# --- Voxel model generation ---

def generate_voxel_solid(
    rng: np.random.Generator,
    grid_size: int = 5,
    faces: int = 6,
    concavities: int = 0,
) -> NDArray[np.bool_]:
    """Generate a random 3D voxel solid.

    Starts with a seed voxel and grows by randomly adding face-adjacent voxels.
    Concavities are created by removing interior voxels that don't disconnect
    the shape.

    Args:
        rng: numpy random generator for reproducibility.
        grid_size: size of the voxel grid along each axis.
        faces: target number of filled voxels (controls shape complexity).
        concavities: number of voxels to remove to create concavities.

    Returns:
        3D boolean array of shape (grid_size, grid_size, grid_size).
    """
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    center = grid_size // 2

    # Start with center voxel
    grid[center, center, center] = True
    filled = [(center, center, center)]

    # Grow the solid by adding adjacent voxels
    target_count = max(1, min(faces, grid_size ** 3 - 1))
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    attempts = 0
    max_attempts = target_count * 20
    while len(filled) < target_count and attempts < max_attempts:
        attempts += 1
        # Pick a random existing voxel
        base = filled[rng.integers(0, len(filled))]
        # Pick a random direction
        dx, dy, dz = directions[rng.integers(0, 6)]
        nx, ny, nz = base[0] + dx, base[1] + dy, base[2] + dz
        if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
            if not grid[nx, ny, nz]:
                grid[nx, ny, nz] = True
                filled.append((nx, ny, nz))

    # Add concavities by removing interior voxels
    if concavities > 0 and len(filled) > 2:
        _add_concavities(grid, filled, concavities, rng)

    return grid


def _add_concavities(
    grid: NDArray[np.bool_],
    filled: list[tuple[int, int, int]],
    count: int,
    rng: np.random.Generator,
) -> None:
    """Remove interior voxels to create concavities without disconnecting the shape."""
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    grid_size = grid.shape[0]
    removed = 0
    attempts = 0
    max_attempts = count * 30

    while removed < count and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, len(filled))
        x, y, z = filled[idx]
        if not grid[x, y, z]:
            continue

        # Only remove if the voxel is interior (all 6 neighbors in bounds)
        is_interior = True
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size):
                is_interior = False
                break
            if not grid[nx, ny, nz]:
                is_interior = False
                break

        if not is_interior:
            continue

        # Temporarily remove and check connectivity
        grid[x, y, z] = False
        if _is_connected(grid):
            removed += 1
        else:
            grid[x, y, z] = True


def _is_connected(grid: NDArray[np.bool_]) -> bool:
    """Check if all filled voxels in the grid form a single connected component."""
    coords = np.argwhere(grid)
    if len(coords) <= 1:
        return True

    # BFS from first filled voxel
    start = tuple(coords[0])
    visited = set()
    visited.add(start)
    queue = [start]
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    grid_size = grid.shape[0]

    while queue:
        x, y, z = queue.pop(0)
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size
                    and grid[nx, ny, nz] and (nx, ny, nz) not in visited):
                visited.add((nx, ny, nz))
                queue.append((nx, ny, nz))

    return len(visited) == int(grid.sum())


# --- Orthographic projection ---

def project_orthographic(
    grid: NDArray[np.bool_],
    axis: str,
) -> NDArray[np.bool_]:
    """Compute 2D orthographic projection of a voxel grid along the specified axis.

    Args:
        grid: 3D boolean voxel grid.
        axis: one of 'front' (along Z), 'top' (along Y), 'side' (along X).

    Returns:
        2D boolean array representing the silhouette.
    """
    if axis == "front":
        # Project along Z axis -> XY plane (collapse Z)
        return grid.any(axis=2)
    elif axis == "top":
        # Project along Y axis -> XZ plane (collapse Y)
        return grid.any(axis=1)
    elif axis == "side":
        # Project along X axis -> YZ plane (collapse X)
        return grid.any(axis=0)
    else:
        raise ValueError(f"Unknown projection axis: {axis!r}. Use 'front', 'top', or 'side'.")


def projections_match(
    proj_a: NDArray[np.bool_],
    proj_b: NDArray[np.bool_],
) -> bool:
    """Check if two 2D projections are identical."""
    if proj_a.shape != proj_b.shape:
        return False
    return bool(np.array_equal(proj_a, proj_b))


# --- SVG rendering ---

# Isometric projection constants
_ISO_ANGLE = np.pi / 6  # 30 degrees
_COS_A = np.cos(_ISO_ANGLE)
_SIN_A = np.sin(_ISO_ANGLE)


def _iso_project(x: float, y: float, z: float) -> tuple[float, float]:
    """Project a 3D point to 2D isometric coordinates.

    Uses standard isometric projection (30-degree angles).
    """
    screen_x = (x - z) * _COS_A
    screen_y = (x + z) * _SIN_A - y
    return screen_x, screen_y


def render_isometric_svg(
    grid: NDArray[np.bool_],
    canvas_size: int = 400,
    cell_size: float = 20.0,
    axis_indicator: str | None = None,
) -> str:
    """Render a voxel grid as an isometric SVG.

    Args:
        grid: 3D boolean voxel grid.
        canvas_size: SVG canvas width and height in pixels.
        cell_size: size of each voxel cube edge in screen pixels.
        axis_indicator: if set, draw an arrow indicating projection axis ('front','top','side').

    Returns:
        SVG string.
    """
    canvas = renderer.create_canvas(canvas_size, canvas_size)
    gs = grid.shape[0]

    # Compute offset to center the isometric view
    cx, cy = canvas_size / 2, canvas_size / 2
    # Offset so the model is centered
    offset_x = cx
    offset_y = cy + (gs * cell_size * _SIN_A) / 2

    # Colors for the three visible faces of a cube
    top_color = "#B8D4E3"
    left_color = "#7BA7C2"
    right_color = "#5B8FAF"
    outline = renderer.STYLE["line_color"]

    # Draw voxels in painter's order: back to front, bottom to top
    for y_idx in range(gs):
        for z_idx in range(gs - 1, -1, -1):
            for x_idx in range(gs):
                if not grid[x_idx, y_idx, z_idx]:
                    continue

                x, y, z = x_idx, y_idx, z_idx

                # Check if this voxel is visible (not fully occluded)
                # For simplicity, always draw - overlapping handled by painter's order

                # 8 corners of the cube
                corners_3d = [
                    (x, y, z),
                    (x + 1, y, z),
                    (x + 1, y, z + 1),
                    (x, y, z + 1),
                    (x, y + 1, z),
                    (x + 1, y + 1, z),
                    (x + 1, y + 1, z + 1),
                    (x, y + 1, z + 1),
                ]
                pts = [
                    (
                        offset_x + px * cell_size * _COS_A - pz * cell_size * _COS_A,
                        offset_y + (px * cell_size * _SIN_A + pz * cell_size * _SIN_A) - py * cell_size,
                    )
                    for px, py, pz in corners_3d
                ]

                # Top face: corners 4,5,6,7 (y+1 face)
                _draw_polygon(canvas, [pts[4], pts[5], pts[6], pts[7]], top_color, outline)
                # Left face: corners 0,3,7,4 (x=x face)
                _draw_polygon(canvas, [pts[0], pts[3], pts[7], pts[4]], left_color, outline)
                # Right face: corners 1,5,6,2 (z+1 face or x+1)
                _draw_polygon(canvas, [pts[1], pts[5], pts[6], pts[2]], right_color, outline)

    # Draw axis indicator
    if axis_indicator:
        _draw_axis_indicator(canvas, axis_indicator, canvas_size)

    return renderer.svg_to_string(canvas)


def _draw_polygon(
    canvas: Any,
    points: list[tuple[float, float]],
    fill: str,
    stroke: str,
) -> None:
    """Draw a filled polygon on the SVG canvas."""
    point_strings = [f"{px},{py}" for px, py in points]
    d = "M " + " L ".join(point_strings) + " Z"
    canvas.add(canvas.path(
        d=d,
        fill=fill,
        stroke=stroke,
        stroke_width=1,
    ))


def _draw_axis_indicator(
    canvas: Any,
    axis: str,
    canvas_size: int,
) -> None:
    """Draw an arrow indicating the projection direction."""
    margin = 30
    arrow_len = 40

    if axis == "front":
        # Arrow pointing into the screen (along Z)
        x_start = canvas_size - margin - arrow_len
        y_pos = margin + 10
        renderer.draw_line(canvas, x_start, y_pos, x_start + arrow_len, y_pos,
                           stroke=renderer.STYLE["highlight_color"], stroke_width=3)
        renderer.draw_text(canvas, x_start + arrow_len / 2, y_pos + 18,
                           "Front (Z)", font_size=10, fill=renderer.STYLE["highlight_color"])
    elif axis == "top":
        # Arrow pointing down (along Y)
        x_pos = canvas_size - margin - 20
        y_start = margin
        renderer.draw_line(canvas, x_pos, y_start, x_pos, y_start + arrow_len,
                           stroke=renderer.STYLE["highlight_color"], stroke_width=3)
        renderer.draw_text(canvas, x_pos, y_start + arrow_len + 14,
                           "Top (Y)", font_size=10, fill=renderer.STYLE["highlight_color"])
    elif axis == "side":
        # Arrow pointing right (along X)
        x_start = canvas_size - margin - arrow_len
        y_pos = margin + 10
        renderer.draw_line(canvas, x_start, y_pos, x_start + arrow_len, y_pos,
                           stroke=renderer.STYLE["highlight_color"], stroke_width=3)
        renderer.draw_text(canvas, x_start + arrow_len / 2, y_pos + 18,
                           "Side (X)", font_size=10, fill=renderer.STYLE["highlight_color"])


def render_projection_svg(
    projection: NDArray[np.bool_],
    canvas_size: int = 300,
    cell_size: float = 20.0,
    label: str = "",
) -> str:
    """Render a 2D orthographic projection as an SVG grid.

    Args:
        projection: 2D boolean array.
        canvas_size: SVG canvas size.
        cell_size: size of each cell in pixels.
        label: optional label drawn above the grid.

    Returns:
        SVG string.
    """
    rows, cols = projection.shape
    grid_w = cols * cell_size
    grid_h = rows * cell_size

    canvas = renderer.create_canvas(canvas_size, canvas_size)
    offset_x = (canvas_size - grid_w) / 2
    offset_y = (canvas_size - grid_h) / 2

    if label:
        renderer.draw_text(canvas, canvas_size / 2, offset_y - 8, label, font_size=12)
        offset_y += 10

    for r in range(rows):
        for c in range(cols):
            x = offset_x + c * cell_size
            y = offset_y + r * cell_size
            fill = "#444444" if projection[r, c] else "#EEEEEE"
            renderer.draw_rect(
                canvas, x, y, cell_size, cell_size,
                fill=fill,
                stroke=renderer.STYLE["grid_color"],
                stroke_width=0.5,
            )

    return renderer.svg_to_string(canvas)


def render_three_projections_svg(
    front: NDArray[np.bool_],
    top: NDArray[np.bool_],
    side: NDArray[np.bool_],
    canvas_size: int = 600,
    cell_size: float = 15.0,
) -> str:
    """Render three orthographic projections in engineering drawing layout.

    Layout:
        [Top]   [   ]
        [Front] [Side]

    Args:
        front: 2D front projection.
        top: 2D top projection.
        side: 2D side projection.
        canvas_size: overall SVG canvas size.
        cell_size: size of each grid cell.

    Returns:
        SVG string.
    """
    canvas = renderer.create_canvas(canvas_size, canvas_size)
    padding = 20
    label_space = 20

    # Compute sub-grid sizes
    max_dim = max(
        front.shape[0], front.shape[1],
        top.shape[0], top.shape[1],
        side.shape[0], side.shape[1],
    )
    sub_size = max_dim * cell_size + 2 * padding

    # Top-left: Top view
    _draw_projection_inset(canvas, top, padding, padding, cell_size, "Top")
    # Bottom-left: Front view
    _draw_projection_inset(canvas, front, padding, padding + sub_size + label_space, cell_size, "Front")
    # Bottom-right: Side view
    _draw_projection_inset(canvas, side, padding + sub_size + label_space, padding + sub_size + label_space, cell_size, "Side")

    return renderer.svg_to_string(canvas)


def _draw_projection_inset(
    canvas: Any,
    projection: NDArray[np.bool_],
    x_offset: float,
    y_offset: float,
    cell_size: float,
    label: str,
) -> None:
    """Draw a single projection grid within a larger canvas."""
    rows, cols = projection.shape

    # Label
    renderer.draw_text(canvas, x_offset + cols * cell_size / 2, y_offset - 4, label, font_size=11)

    for r in range(rows):
        for c in range(cols):
            x = x_offset + c * cell_size
            y = y_offset + r * cell_size
            fill = "#444444" if projection[r, c] else "#EEEEEE"
            renderer.draw_rect(
                canvas, x, y, cell_size, cell_size,
                fill=fill,
                stroke=renderer.STYLE["grid_color"],
                stroke_width=0.5,
            )


def encode_projection(projection: NDArray[np.bool_]) -> str:
    """Encode a 2D projection as a compact string for embedding in SVG metadata."""
    rows, cols = projection.shape
    bits = "".join("1" if projection[r, c] else "0" for r in range(rows) for c in range(cols))
    return f"{rows}x{cols}:{bits}"


def decode_projection(encoded: str) -> NDArray[np.bool_]:
    """Decode a projection string back to a 2D boolean array."""
    shape_str, bits = encoded.split(":")
    rows, cols = (int(x) for x in shape_str.split("x"))
    arr = np.array([b == "1" for b in bits], dtype=bool).reshape(rows, cols)
    return arr


def encode_voxel_grid(grid: NDArray[np.bool_]) -> str:
    """Encode a 3D voxel grid as a compact string for metadata storage."""
    sx, sy, sz = grid.shape
    bits = "".join(
        "1" if grid[x, y, z] else "0"
        for x in range(sx) for y in range(sy) for z in range(sz)
    )
    return f"{sx}x{sy}x{sz}:{bits}"


def decode_voxel_grid(encoded: str) -> NDArray[np.bool_]:
    """Decode a voxel grid string back to a 3D boolean array."""
    shape_str, bits = encoded.split(":")
    sx, sy, sz = (int(x) for x in shape_str.split("x"))
    arr = np.array([b == "1" for b in bits], dtype=bool).reshape(sx, sy, sz)
    return arr
