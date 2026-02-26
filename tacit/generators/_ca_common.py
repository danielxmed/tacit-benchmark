"""Shared cellular automata utilities for forward and inverse generators.

Provides:
- Rule encoding/decoding for outer totalistic 2D CA (Moore neighborhood)
- Grid simulation (step function)
- SVG rendering of grids and rule tables
- Grid parsing from SVG strings
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tacit.core.renderer import (
    STYLE,
    create_canvas,
    draw_rect,
    draw_text,
    svg_to_string,
)


# ---------------------------------------------------------------------------
# Color palette for CA states (index = state value)
# ---------------------------------------------------------------------------
# State 0 is always white (dead/empty); states 1..N use STYLE["colors"]
def state_color(state: int) -> str:
    """Return the fill color for a given cell state."""
    if state == 0:
        return "#FFFFFF"
    idx = (state - 1) % len(STYLE["colors"])
    return STYLE["colors"][idx]


# ---------------------------------------------------------------------------
# Rule representation
# ---------------------------------------------------------------------------
# An *outer totalistic* rule on a 2D Moore neighborhood maps
#   (current_state, neighbor_sum) -> next_state
# where neighbor_sum is the sum of the 8 Moore neighbors' state values.
#
# For *num_states* states on a Moore neighborhood, the max neighbor sum
# is 8 * (num_states - 1).
#
# The rule is stored as a 2-D numpy array of shape
#   (num_states, max_neighbor_sum + 1)
# where rule[s][n] = next state for a cell in state *s* whose Moore
# neighbor sum is *n*.
# ---------------------------------------------------------------------------


def max_neighbor_sum(num_states: int) -> int:
    """Maximum possible Moore-neighborhood sum for the given state count."""
    return 8 * (num_states - 1)


def generate_rule(
    num_states: int, rng: np.random.Generator
) -> NDArray[np.int32]:
    """Generate a random outer-totalistic rule table.

    Returns:
        Array of shape (num_states, max_neighbor_sum + 1) with values in
        [0, num_states).
    """
    max_ns = max_neighbor_sum(num_states)
    return rng.integers(0, num_states, size=(num_states, max_ns + 1), dtype=np.int32)


def generate_initial_grid(
    size: int, num_states: int, rng: np.random.Generator
) -> NDArray[np.int32]:
    """Generate a random initial grid."""
    return rng.integers(0, num_states, size=(size, size), dtype=np.int32)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _neighbor_sum(grid: NDArray[np.int32]) -> NDArray[np.int32]:
    """Compute Moore-neighborhood sum for every cell using wrapping."""
    rows, cols = grid.shape
    total = np.zeros_like(grid)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += np.roll(np.roll(grid, -dr, axis=0), -dc, axis=1)
    return total


def step(
    grid: NDArray[np.int32], rule: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Advance the grid by one time step under the given rule.

    Uses wrapping (toroidal) boundary conditions.
    """
    ns = _neighbor_sum(grid)
    new_grid = np.empty_like(grid)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            s = int(grid[r, c])
            n = int(ns[r, c])
            # Clamp neighbor sum to rule table bounds
            n = min(n, rule.shape[1] - 1)
            new_grid[r, c] = rule[s, n]
    return new_grid


def simulate(
    grid: NDArray[np.int32], rule: NDArray[np.int32], steps: int
) -> NDArray[np.int32]:
    """Simulate *steps* time steps, returning the final grid."""
    current = grid.copy()
    for _ in range(steps):
        current = step(current, rule)
    return current


# ---------------------------------------------------------------------------
# SVG rendering — grids
# ---------------------------------------------------------------------------
CELL_SIZE = 24
GRID_PADDING = 20
LABEL_HEIGHT = 24


def render_grid_svg(
    grid: NDArray[np.int32],
    title: str = "",
    x_offset: float = 0,
    y_offset: float = 0,
    cell_size: int = CELL_SIZE,
) -> tuple[list[dict[str, Any]], float, float]:
    """Produce drawing instructions for a grid (not a full SVG).

    Returns:
        (elements, width, height) where elements is a list of dicts
        describing rects and texts to draw.
    """
    rows, cols = grid.shape
    elements: list[dict[str, Any]] = []
    title_h = LABEL_HEIGHT if title else 0

    for r in range(rows):
        for c in range(cols):
            elements.append({
                "type": "rect",
                "x": x_offset + c * cell_size,
                "y": y_offset + title_h + r * cell_size,
                "w": cell_size,
                "h": cell_size,
                "fill": state_color(int(grid[r, c])),
            })
    if title:
        elements.append({
            "type": "text",
            "x": x_offset + cols * cell_size / 2,
            "y": y_offset + 16,
            "text": title,
        })
    width = cols * cell_size
    height = title_h + rows * cell_size
    return elements, width, height


def draw_elements(canvas: Any, elements: list[dict[str, Any]]) -> None:
    """Draw a list of element dicts onto an svgwrite canvas."""
    for el in elements:
        if el["type"] == "rect":
            draw_rect(
                canvas,
                el["x"], el["y"], el["w"], el["h"],
                fill=el["fill"],
                stroke=STYLE["grid_color"],
                stroke_width=1,
            )
        elif el["type"] == "text":
            draw_text(canvas, el["x"], el["y"], el["text"], font_size=12)


def grid_to_svg(
    grid: NDArray[np.int32],
    title: str = "",
    cell_size: int = CELL_SIZE,
) -> str:
    """Render a single grid to a complete SVG string."""
    rows, cols = grid.shape
    title_h = LABEL_HEIGHT if title else 0
    w = cols * cell_size + 2 * GRID_PADDING
    h = title_h + rows * cell_size + 2 * GRID_PADDING
    canvas = create_canvas(int(w), int(h))
    elems, _, _ = render_grid_svg(grid, title, GRID_PADDING, GRID_PADDING, cell_size)
    draw_elements(canvas, elems)
    return svg_to_string(canvas)


# ---------------------------------------------------------------------------
# SVG rendering — rule table
# ---------------------------------------------------------------------------

def render_rule_table_svg(
    rule: NDArray[np.int32],
    title: str = "Rule Table",
    cell_size: int = 16,
) -> str:
    """Render a rule table as an SVG string.

    Rows = current state, columns = neighbor sum.
    Each cell is colored according to the output state.
    Row and column headers are included.
    """
    num_states, num_sums = rule.shape
    header = 30  # space for column headers
    row_header = 40  # space for row headers
    padding = GRID_PADDING
    title_h = LABEL_HEIGHT

    total_w = row_header + num_sums * cell_size + 2 * padding
    total_h = title_h + header + num_states * cell_size + 2 * padding

    canvas = create_canvas(int(total_w), int(total_h))

    # Title
    draw_text(canvas, total_w / 2, padding + 14, title, font_size=12)

    base_x = padding + row_header
    base_y = padding + title_h + header

    # Column headers (neighbor sums) — only show every few to avoid clutter
    step_label = max(1, num_sums // 10)
    for j in range(0, num_sums, step_label):
        draw_text(
            canvas,
            base_x + j * cell_size + cell_size / 2,
            base_y - 6,
            str(j),
            font_size=8,
        )

    # Row headers (current state)
    for i in range(num_states):
        draw_text(
            canvas,
            padding + row_header / 2,
            base_y + i * cell_size + cell_size / 2 + 4,
            f"s{i}",
            font_size=9,
        )

    # Cells
    for i in range(num_states):
        for j in range(num_sums):
            draw_rect(
                canvas,
                base_x + j * cell_size,
                base_y + i * cell_size,
                cell_size,
                cell_size,
                fill=state_color(int(rule[i, j])),
                stroke=STYLE["grid_color"],
                stroke_width=0.5,
            )

    return svg_to_string(canvas)


# ---------------------------------------------------------------------------
# SVG rendering — combined views
# ---------------------------------------------------------------------------

def render_forward_puzzle_svg(
    initial_grid: NDArray[np.int32],
    rule: NDArray[np.int32],
    steps: int,
    cell_size: int = CELL_SIZE,
) -> str:
    """Render the forward puzzle: initial grid + rule table.

    Shows the initial grid on the left and the rule table on the right,
    with a label indicating how many steps to simulate.
    """
    rows, cols = initial_grid.shape
    padding = GRID_PADDING

    # Grid section
    grid_elems, gw, gh = render_grid_svg(
        initial_grid, f"State T (grid {rows}x{cols})",
        padding, padding + LABEL_HEIGHT, cell_size,
    )

    # Rule table section — rendered separately then embedded
    rule_svg_str = render_rule_table_svg(rule, "Transition Rule", cell_size=12)

    # We compose both into one SVG by drawing the grid and then embedding
    # the rule description as text/visual below.
    num_states, num_sums = rule.shape
    rule_cell = 12
    rule_w = 40 + num_sums * rule_cell + 2 * padding
    rule_h = LABEL_HEIGHT + 30 + num_states * rule_cell + 2 * padding

    total_w = padding + gw + padding + max(int(rule_w), 100) + padding
    top_label = LABEL_HEIGHT + 10
    total_h = max(int(gh), int(rule_h)) + top_label + 2 * padding + LABEL_HEIGHT

    canvas = create_canvas(int(total_w), int(total_h))

    # Title
    draw_text(canvas, total_w / 2, 18, f"Predict state after {steps} step(s)", font_size=14)

    # Draw initial grid
    grid_elems2, _, _ = render_grid_svg(
        initial_grid, f"State T",
        padding, top_label + padding, cell_size,
    )
    draw_elements(canvas, grid_elems2)

    # Draw rule table to the right of the grid
    rule_x = padding + gw + padding * 2
    rule_y = top_label + padding

    # Row/column headers and cells for rule table
    header_h = 30
    row_header_w = 40

    draw_text(
        canvas,
        rule_x + rule_w / 2,
        rule_y + 14,
        "Transition Rule",
        font_size=12,
    )

    rule_base_x = rule_x + row_header_w
    rule_base_y = rule_y + LABEL_HEIGHT + header_h

    step_label = max(1, num_sums // 10)
    for j in range(0, num_sums, step_label):
        draw_text(
            canvas,
            rule_base_x + j * rule_cell + rule_cell / 2,
            rule_base_y - 6,
            str(j),
            font_size=7,
        )

    for i in range(num_states):
        draw_text(
            canvas,
            rule_x + row_header_w / 2,
            rule_base_y + i * rule_cell + rule_cell / 2 + 3,
            f"s{i}",
            font_size=8,
        )

    for i in range(num_states):
        for j in range(num_sums):
            draw_rect(
                canvas,
                rule_base_x + j * rule_cell,
                rule_base_y + i * rule_cell,
                rule_cell,
                rule_cell,
                fill=state_color(int(rule[i, j])),
                stroke=STYLE["grid_color"],
                stroke_width=0.5,
            )

    return svg_to_string(canvas)


def render_inverse_puzzle_svg(
    grid_t: NDArray[np.int32],
    grid_tk: NDArray[np.int32],
    steps: int,
    cell_size: int = CELL_SIZE,
) -> str:
    """Render the inverse puzzle: state T and state T+k grids side by side."""
    rows, cols = grid_t.shape
    padding = GRID_PADDING

    grid_w = cols * cell_size
    grid_h = LABEL_HEIGHT + rows * cell_size
    arrow_space = 60

    total_w = 2 * padding + grid_w + arrow_space + grid_w + padding
    top_label = LABEL_HEIGHT + 10
    total_h = top_label + grid_h + 2 * padding

    canvas = create_canvas(int(total_w), int(total_h))

    # Title
    draw_text(
        canvas, total_w / 2, 18,
        f"Infer the rule ({steps} step(s))",
        font_size=14,
    )

    # State T grid
    elems_t, _, _ = render_grid_svg(
        grid_t, "State T",
        padding, top_label, cell_size,
    )
    draw_elements(canvas, elems_t)

    # Arrow
    arrow_x = padding + grid_w + arrow_space / 2
    arrow_y = top_label + grid_h / 2
    draw_text(canvas, arrow_x, arrow_y, "-->", font_size=16)
    draw_text(canvas, arrow_x, arrow_y + 18, f"{steps} step(s)", font_size=9)

    # State T+k grid
    elems_tk, _, _ = render_grid_svg(
        grid_tk, f"State T+{steps}",
        padding + grid_w + arrow_space, top_label, cell_size,
    )
    draw_elements(canvas, elems_tk)

    return svg_to_string(canvas)


# ---------------------------------------------------------------------------
# SVG parsing — extract rect attributes (attribute-order agnostic)
# ---------------------------------------------------------------------------

_ATTR_RE = re.compile(r'(\w[\w-]*)="([^"]*)"')
_RECT_RE = re.compile(r'<rect\b([^>]*)/?>', re.DOTALL)


def _parse_rects(svg: str) -> list[dict[str, str]]:
    """Extract all <rect> elements as dicts of attribute -> value.

    This is attribute-order agnostic, which is critical because
    svgwrite may emit attributes in any order.
    """
    results: list[dict[str, str]] = []
    for m in _RECT_RE.finditer(svg):
        attrs = dict(_ATTR_RE.findall(m.group(1)))
        results.append(attrs)
    return results


def _hex_to_state_map() -> dict[str, int]:
    """Build a reverse map from hex color -> state index."""
    mapping: dict[str, int] = {"#FFFFFF": 0, "#ffffff": 0}
    for i, color in enumerate(STYLE["colors"]):
        mapping[color.upper()] = i + 1
        mapping[color.lower()] = i + 1
    return mapping


def parse_grid_from_svg(svg: str, grid_size: int) -> NDArray[np.int32] | None:
    """Extract a grid from an SVG string by reading rect fill colors.

    Expects the grid to be rendered as rect elements with the standard
    state_color palette. Returns None if parsing fails.
    """
    color_map = _hex_to_state_map()
    all_rects = _parse_rects(svg)

    rects: list[tuple[float, float, str]] = []
    for attrs in all_rects:
        x_s = attrs.get("x")
        y_s = attrs.get("y")
        fill = attrs.get("fill", "")
        w_s = attrs.get("width")
        h_s = attrs.get("height")

        if x_s is None or y_s is None or not fill:
            continue

        x_f, y_f = float(x_s), float(y_s)
        fill_upper = fill.upper().strip()

        # Skip the full-canvas background rect
        if fill_upper in ("#FFFFFF", STYLE["background"].upper()):
            if w_s and float(w_s) > grid_size * CELL_SIZE:
                continue

        if fill_upper in color_map or fill.strip() in color_map:
            rects.append((x_f, y_f, fill.strip()))

    if not rects:
        return None

    # Deduplicate by (x, y)
    seen: set[tuple[float, float]] = set()
    unique_rects: list[tuple[float, float, str]] = []
    for x, y, fill in rects:
        key = (round(x, 2), round(y, 2))
        if key not in seen:
            seen.add(key)
            unique_rects.append((x, y, fill))

    rects = unique_rects
    rects.sort(key=lambda t: (t[1], t[0]))

    xs = sorted(set(round(r[0], 2) for r in rects))
    ys = sorted(set(round(r[1], 2) for r in rects))

    detected_rows = len(ys)
    detected_cols = len(xs)

    if detected_rows != grid_size or detected_cols != grid_size:
        if detected_rows >= grid_size and detected_cols >= grid_size:
            ys = ys[:grid_size]
            xs = xs[:grid_size]
        else:
            return None

    x_to_col = {round(v, 2): i for i, v in enumerate(xs[:grid_size])}
    y_to_row = {round(v, 2): i for i, v in enumerate(ys[:grid_size])}

    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    for x, y, fill in rects:
        rx, ry = round(x, 2), round(y, 2)
        if rx in x_to_col and ry in y_to_row:
            col = x_to_col[rx]
            row = y_to_row[ry]
            color_key = fill.upper() if fill.upper() in color_map else fill
            if color_key in color_map:
                grid[row, col] = color_map[color_key]

    return grid


def parse_rule_from_svg(
    svg: str, num_states: int
) -> NDArray[np.int32] | None:
    """Extract a rule table from an SVG string.

    The rule table is rendered as a grid of small colored cells. We parse
    rect elements and reconstruct the 2D rule array.

    Returns None if parsing fails.
    """
    color_map = _hex_to_state_map()
    max_ns = max_neighbor_sum(num_states)
    expected_cols = max_ns + 1

    all_rects = _parse_rects(svg)

    rects: list[tuple[float, float, float, str]] = []
    for attrs in all_rects:
        x_s = attrs.get("x")
        y_s = attrs.get("y")
        w_s = attrs.get("width")
        h_s = attrs.get("height")
        fill = attrs.get("fill", "")
        stroke = attrs.get("stroke", "")

        if x_s is None or y_s is None or w_s is None or not fill:
            continue

        w_f = float(w_s)
        h_f = float(h_s) if h_s else w_f

        # Rule cells are small squares; skip large rects
        if w_f > 20 or h_f > 20:
            continue

        # Skip background rects (no stroke = canvas background, not a data cell)
        if not stroke:
            continue

        fill_upper = fill.upper().strip()
        if fill_upper in color_map or fill.strip() in color_map:
            rects.append((float(x_s), float(y_s), w_f, fill.strip()))

    if not rects:
        return None

    # Deduplicate
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float, float, str]] = []
    for x, y, w, fill in rects:
        key = (round(x, 2), round(y, 2))
        if key not in seen:
            seen.add(key)
            unique.append((x, y, w, fill))
    rects = unique

    rects.sort(key=lambda t: (t[1], t[0]))

    xs = sorted(set(round(r[0], 2) for r in rects))
    ys = sorted(set(round(r[1], 2) for r in rects))

    if len(ys) < num_states or len(xs) < expected_cols:
        return None

    xs = xs[:expected_cols]
    ys = ys[:num_states]

    x_to_col = {round(v, 2): i for i, v in enumerate(xs)}
    y_to_row = {round(v, 2): i for i, v in enumerate(ys)}

    rule = np.zeros((num_states, expected_cols), dtype=np.int32)
    for x, y, w, fill in rects:
        rx, ry = round(x, 2), round(y, 2)
        if rx in x_to_col and ry in y_to_row:
            col = x_to_col[rx]
            row = y_to_row[ry]
            color_key = fill.upper() if fill.upper() in color_map else fill
            if color_key in color_map:
                rule[row, col] = color_map[color_key]

    return rule


# ---------------------------------------------------------------------------
# PNG parsing — extract grid/rule by sampling pixel colors at cell centers
# ---------------------------------------------------------------------------


def _build_state_palette() -> dict[str, int]:
    """Build hex_color -> state_index palette from STYLE colors."""
    palette: dict[str, int] = {"#FFFFFF": 0}
    for i, color in enumerate(STYLE["colors"]):
        palette[color.upper()] = i + 1
    return palette


def parse_grid_from_png(
    png_bytes: bytes,
    grid_size: int,
    cell_size: int = CELL_SIZE,
    padding: int = GRID_PADDING,
    title_offset: int = LABEL_HEIGHT,
) -> NDArray[np.int32] | None:
    """Extract a grid from PNG by sampling cell-center colors.

    Assumes the PNG was rasterized at the SVG's natural viewport size
    (1:1 pixel mapping with rendering coordinates).
    """
    from tacit.core.cv_utils import (
        find_closest_palette_color,
        png_to_numpy,
        sample_color,
    )

    img = png_to_numpy(png_bytes)
    palette = _build_state_palette()
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)

    for r in range(grid_size):
        for c in range(grid_size):
            cx = int(padding + c * cell_size + cell_size / 2)
            cy = int(padding + title_offset + r * cell_size + cell_size / 2)
            if cy >= img.shape[0] or cx >= img.shape[1]:
                return None
            pixel = sample_color(img, cx, cy)
            state = find_closest_palette_color(pixel, palette)
            if state is None:
                return None
            grid[r, c] = state

    return grid


def parse_rule_from_png(
    png_bytes: bytes,
    num_states: int,
    cell_size: int = 12,
    row_header_w: int = 40,
    padding: int = GRID_PADDING,
    title_offset: int = LABEL_HEIGHT,
    header_h: int = 30,
) -> NDArray[np.int32] | None:
    """Extract a rule table from PNG by sampling cell-center colors.

    Layout mirrors render_rule_table_svg():
    - Title at top
    - Column headers
    - Row headers on left
    - Grid of colored cells
    """
    from tacit.core.cv_utils import (
        find_closest_palette_color,
        png_to_numpy,
        sample_color,
    )

    img = png_to_numpy(png_bytes)
    palette = _build_state_palette()
    max_ns = max_neighbor_sum(num_states)
    expected_cols = max_ns + 1

    rule = np.zeros((num_states, expected_cols), dtype=np.int32)

    base_x = padding + row_header_w
    base_y = padding + title_offset + header_h

    for i in range(num_states):
        for j in range(expected_cols):
            cx = int(base_x + j * cell_size + cell_size / 2)
            cy = int(base_y + i * cell_size + cell_size / 2)
            if cy >= img.shape[0] or cx >= img.shape[1]:
                return None
            pixel = sample_color(img, cx, cy)
            state = find_closest_palette_color(pixel, palette)
            if state is None:
                return None
            rule[i, j] = state

    return rule
