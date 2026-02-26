"""Raven's Progressive Matrices generator for TACIT Benchmark.

Generates 3x3 grids where each cell contains a tile with attributes
(shape, color, size, rotation, count). Transformation rules govern how
attributes progress across rows and columns. The bottom-right cell is
missing and must be inferred.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tacit.core.renderer import (
    STYLE,
    create_canvas,
    draw_circle,
    draw_path,
    draw_rect,
    draw_text,
    svg_to_string,
)
from tacit.core.types import (
    DifficultyParams,
    DifficultyRange,
    PuzzleInstance,
    VerificationResult,
)
from tacit.generators.base import BaseGenerator

# --- Constants ---

SHAPES = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon"]
SIZES = ["small", "medium", "large"]
ROTATIONS = [0, 90, 180, 270]
COUNTS = [1, 2, 3, 4]

# Use the STYLE color palette from renderer
COLORS = STYLE["colors"]

# Rule types
RULE_ATTRIBUTES = ["shape", "color", "rotation", "count"]


@dataclass
class TileAttributes:
    """Attributes for a single tile in the Raven's matrix."""

    shape: str = "circle"
    color: str = COLORS[0]
    size: str = "medium"
    rotation: int = 0
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "color": self.color,
            "size": self.size,
            "rotation": self.rotation,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TileAttributes:
        return cls(
            shape=d["shape"],
            color=d["color"],
            size=d["size"],
            rotation=d["rotation"],
            count=d["count"],
        )


@dataclass
class RavenRule:
    """A single transformation rule for the matrix."""

    attribute: str  # which attribute this rule governs
    rule_type: str  # "additive" or "compositional"
    values: list[Any] = field(default_factory=list)  # the 3 values for row/col


@dataclass
class RavenPuzzleData:
    """Internal representation of a Raven's matrix puzzle."""

    grid: list[list[TileAttributes]]  # 3x3 grid of tile attributes
    rules: list[RavenRule]
    answer_tile: TileAttributes  # the correct bottom-right tile


# --- Shape drawing helpers ---

def _shape_path_d(
    shape: str, cx: float, cy: float, r: float, rotation: int
) -> str | None:
    """Return SVG path d-string for a polygon shape, or None for circle/square."""
    angle_offset = math.radians(rotation)

    if shape == "triangle":
        pts = []
        for i in range(3):
            a = angle_offset + math.radians(i * 120 - 90)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d = f"M {pts[0][0]},{pts[0][1]}"
        for p in pts[1:]:
            d += f" L {p[0]},{p[1]}"
        d += " Z"
        return d

    if shape == "diamond":
        pts = []
        for i in range(4):
            a = angle_offset + math.radians(i * 90)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d = f"M {pts[0][0]},{pts[0][1]}"
        for p in pts[1:]:
            d += f" L {p[0]},{p[1]}"
        d += " Z"
        return d

    if shape == "pentagon":
        pts = []
        for i in range(5):
            a = angle_offset + math.radians(i * 72 - 90)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d = f"M {pts[0][0]},{pts[0][1]}"
        for p in pts[1:]:
            d += f" L {p[0]},{p[1]}"
        d += " Z"
        return d

    if shape == "hexagon":
        pts = []
        for i in range(6):
            a = angle_offset + math.radians(i * 60 - 90)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        d = f"M {pts[0][0]},{pts[0][1]}"
        for p in pts[1:]:
            d += f" L {p[0]},{p[1]}"
        d += " Z"
        return d

    return None


def _size_to_radius(size: str, cell_size: float) -> float:
    """Convert size string to radius relative to cell size."""
    ratios = {"small": 0.15, "medium": 0.25, "large": 0.35}
    return cell_size * ratios.get(size, 0.25)


def _draw_tile_on_canvas(
    canvas: Any,
    tile: TileAttributes,
    cx: float,
    cy: float,
    cell_size: float,
) -> None:
    """Draw a tile's shapes on the canvas at the given center position."""
    r = _size_to_radius(tile.size, cell_size)

    # Position multiple instances (count) in a grid-like arrangement
    offsets = _count_offsets(tile.count, r)

    for ox, oy in offsets:
        sx = cx + ox
        sy = cy + oy
        _draw_single_shape(canvas, tile.shape, sx, sy, r, tile.rotation, tile.color)


def _count_offsets(count: int, r: float) -> list[tuple[float, float]]:
    """Calculate offsets for multiple shape instances."""
    spacing = r * 1.5
    if count == 1:
        return [(0, 0)]
    if count == 2:
        return [(-spacing / 2, 0), (spacing / 2, 0)]
    if count == 3:
        return [(-spacing / 2, -spacing / 3), (spacing / 2, -spacing / 3), (0, spacing / 2)]
    # count == 4
    return [
        (-spacing / 2, -spacing / 2),
        (spacing / 2, -spacing / 2),
        (-spacing / 2, spacing / 2),
        (spacing / 2, spacing / 2),
    ]


def _draw_single_shape(
    canvas: Any,
    shape: str,
    cx: float,
    cy: float,
    r: float,
    rotation: int,
    color: str,
) -> None:
    """Draw a single shape primitive on the canvas."""
    if shape == "circle":
        draw_circle(canvas, cx, cy, r, fill=color)
    elif shape == "square":
        # Draw square as a rotated rectangle
        half = r * 0.8
        if rotation == 0 or rotation == 180:
            draw_rect(canvas, cx - half, cy - half, half * 2, half * 2, fill=color)
        else:
            # For rotated squares, draw as a diamond-like path
            d = _shape_path_d("diamond", cx, cy, r, rotation=0)
            if d:
                draw_path(canvas, d, fill=color)
    else:
        d = _shape_path_d(shape, cx, cy, r, rotation)
        if d:
            draw_path(canvas, d, fill=color)


def _render_tile_svg(tile: TileAttributes, width: int = 100, height: int = 100) -> str:
    """Render a single tile to SVG string."""
    canvas = create_canvas(width, height)
    cx = width / 2
    cy = height / 2
    _draw_tile_on_canvas(canvas, tile, cx, cy, min(width, height))
    return svg_to_string(canvas)


# --- Rule generation and application ---

def _generate_rules(
    num_rules: int,
    complexity: str,
    rng: np.random.Generator,
) -> list[RavenRule]:
    """Generate transformation rules for the matrix.

    Args:
        num_rules: Number of rules to generate (1-4).
        complexity: "additive" or "compositional".
        rng: Random number generator.

    Returns:
        List of RavenRule objects.
    """
    # Pick which attributes get rules
    available = list(RULE_ATTRIBUTES)
    rng.shuffle(available)
    chosen_attrs = available[:num_rules]

    rules: list[RavenRule] = []
    for attr in chosen_attrs:
        if attr == "shape":
            shapes_pool = list(SHAPES)
            rng.shuffle(shapes_pool)
            values = shapes_pool[:3]
        elif attr == "color":
            color_indices = rng.choice(len(COLORS), size=3, replace=False)
            values = [COLORS[int(i)] for i in color_indices]
        elif attr == "rotation":
            # Pick 3 distinct rotations
            rot_pool = list(ROTATIONS)
            rng.shuffle(rot_pool)
            values = [int(r) for r in rot_pool[:3]]
        elif attr == "count":
            count_pool = list(COUNTS)
            rng.shuffle(count_pool)
            values = [int(c) for c in count_pool[:3]]
        else:
            continue

        rules.append(RavenRule(
            attribute=attr,
            rule_type=complexity,
            values=values,
        ))

    return rules


def _apply_rules(
    rules: list[RavenRule],
    row: int,
    col: int,
    base_tile: TileAttributes,
) -> TileAttributes:
    """Apply rules to determine tile attributes at (row, col).

    For additive rules: attribute is determined by column index.
    For compositional rules: attribute is determined by (row + col) % 3.
    """
    attrs = base_tile.to_dict()

    for rule in rules:
        if rule.rule_type == "additive":
            idx = col % len(rule.values)
        else:  # compositional
            idx = (row + col) % len(rule.values)

        attrs[rule.attribute] = rule.values[idx]

    return TileAttributes.from_dict(attrs)


def _build_grid(
    rules: list[RavenRule],
    base_tile: TileAttributes,
) -> list[list[TileAttributes]]:
    """Build the 3x3 grid by applying rules to each cell."""
    grid: list[list[TileAttributes]] = []
    for row in range(3):
        row_tiles: list[TileAttributes] = []
        for col in range(3):
            tile = _apply_rules(rules, row, col, base_tile)
            row_tiles.append(tile)
        grid.append(row_tiles)
    return grid


# --- SVG encoding/decoding for verification ---

_ATTR_PREFIX = "data-tacit-"


def _encode_tile_attrs(tile: TileAttributes) -> str:
    """Encode tile attributes as data attributes in an SVG group marker."""
    attrs = tile.to_dict()
    parts = [f'{_ATTR_PREFIX}{k}="{v}"' for k, v in attrs.items()]
    return " ".join(parts)


def _extract_tile_attrs(svg_string: str) -> TileAttributes | None:
    """Extract tile attributes from data attributes embedded in SVG."""
    import re

    attrs: dict[str, Any] = {}
    for key in ["shape", "color", "size", "rotation", "count"]:
        pattern = rf'{_ATTR_PREFIX}{key}="([^"]*)"'
        match = re.search(pattern, svg_string)
        if match:
            val = match.group(1)
            if key == "rotation":
                val = int(val)
            elif key == "count":
                val = int(val)
            attrs[key] = val

    if not attrs:
        return None

    # Fill defaults for any missing attributes
    defaults = TileAttributes().to_dict()
    for k, v in defaults.items():
        if k not in attrs:
            attrs[k] = v

    return TileAttributes.from_dict(attrs)


# --- Main Generator ---

class RavenGenerator(BaseGenerator):
    """Generator for Raven's Progressive Matrices puzzles.

    Creates 3x3 grids with transformation rules governing shape, color,
    rotation, and count attributes. The bottom-right cell is the answer.
    """

    def __init__(self) -> None:
        super().__init__(task_name="raven")

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[RavenPuzzleData, TileAttributes]:
        """Generate a Raven's matrix puzzle.

        Returns:
            (puzzle_data, solution_data) where solution_data is the answer tile.
        """
        num_rules = difficulty.params.get("rules", 1)
        complexity = difficulty.params.get("complexity", "additive")

        # Generate base tile (defaults for attributes not governed by rules)
        base_shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
        base_color = COLORS[int(rng.integers(0, len(COLORS)))]
        base_size = SIZES[int(rng.integers(0, len(SIZES)))]
        base_rotation = int(ROTATIONS[int(rng.integers(0, len(ROTATIONS)))])
        base_count = int(COUNTS[int(rng.integers(0, len(COUNTS)))])

        base_tile = TileAttributes(
            shape=base_shape,
            color=base_color,
            size=base_size,
            rotation=base_rotation,
            count=base_count,
        )

        rules = _generate_rules(num_rules, complexity, rng)
        grid = _build_grid(rules, base_tile)
        answer_tile = grid[2][2]

        puzzle_data = RavenPuzzleData(
            grid=grid,
            rules=rules,
            answer_tile=answer_tile,
        )

        return puzzle_data, answer_tile

    def _generate_puzzle_svg(self, puzzle_data: Any) -> str:
        """Render the 3x3 grid with '?' in the bottom-right cell."""
        data: RavenPuzzleData = puzzle_data
        cell_size = 100
        padding = 10
        grid_size = cell_size * 3 + padding * 4
        canvas = create_canvas(grid_size, grid_size)

        for row in range(3):
            for col in range(3):
                x = padding + col * (cell_size + padding)
                y = padding + row * (cell_size + padding)

                # Draw cell border
                draw_rect(
                    canvas, x, y, cell_size, cell_size,
                    fill=STYLE["background"],
                    stroke=STYLE["grid_color"],
                )

                if row == 2 and col == 2:
                    # Missing cell: draw "?"
                    draw_text(
                        canvas,
                        x + cell_size / 2,
                        y + cell_size / 2 + 8,
                        "?",
                        font_size=32,
                        fill=STYLE["highlight_color"],
                    )
                else:
                    tile = data.grid[row][col]
                    cx = x + cell_size / 2
                    cy = y + cell_size / 2
                    _draw_tile_on_canvas(canvas, tile, cx, cy, cell_size)

        return svg_to_string(canvas)

    def _generate_solution_svg(
        self, puzzle_data: Any, solution_data: Any
    ) -> str:
        """Render the solution tile as a standalone SVG with embedded attributes."""
        tile: TileAttributes = solution_data
        width, height = 100, 100
        canvas = create_canvas(width, height)

        _draw_tile_on_canvas(canvas, tile, width / 2, height / 2, min(width, height))

        # Embed tile attributes as data attributes for verification
        svg_str = svg_to_string(canvas)
        marker = f"<!-- {_encode_tile_attrs(tile)} -->"
        # Insert marker right after the opening <svg tag's closing >
        svg_str = svg_str.replace("</svg>", f"{marker}</svg>")
        return svg_str

    def _generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a distractor by altering one attribute of the solution."""
        tile: TileAttributes = solution_data
        attrs = tile.to_dict()

        if violation_type == "wrong_shape":
            others = [s for s in SHAPES if s != attrs["shape"]]
            attrs["shape"] = others[int(rng.integers(0, len(others)))]

        elif violation_type == "wrong_color":
            others = [c for c in COLORS if c != attrs["color"]]
            attrs["color"] = others[int(rng.integers(0, len(others)))]

        elif violation_type == "wrong_rotation":
            others = [r for r in ROTATIONS if r != attrs["rotation"]]
            attrs["rotation"] = int(others[int(rng.integers(0, len(others)))])

        elif violation_type == "wrong_count":
            others = [c for c in COUNTS if c != attrs["count"]]
            attrs["count"] = int(others[int(rng.integers(0, len(others)))])

        distractor_tile = TileAttributes.from_dict(attrs)

        width, height = 100, 100
        canvas = create_canvas(width, height)
        _draw_tile_on_canvas(
            canvas, distractor_tile, width / 2, height / 2, min(width, height)
        )
        svg_str = svg_to_string(canvas)
        marker = f"<!-- {_encode_tile_attrs(distractor_tile)} -->"
        svg_str = svg_str.replace("</svg>", f"{marker}</svg>")

        return svg_str, violation_type

    def _available_violations(self) -> list[str]:
        """List available distractor violation types."""
        return ["wrong_shape", "wrong_color", "wrong_rotation", "wrong_count"]

    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify a candidate solution against the expected answer.

        Extracts tile attributes from the candidate SVG's embedded data
        and compares them to the expected answer tile attributes.
        """
        # Extract expected answer from puzzle metadata or regenerate
        # We embed attributes in the solution SVG as data-tacit-* comments
        expected = _extract_tile_attrs(puzzle.solution_svg)
        candidate = _extract_tile_attrs(candidate_svg)

        if expected is None:
            return VerificationResult(
                passed=False,
                reason="Could not extract expected tile attributes from solution SVG",
            )

        if candidate is None:
            return VerificationResult(
                passed=False,
                reason="Could not extract tile attributes from candidate SVG",
            )

        # Compare all attributes
        mismatches: dict[str, Any] = {}
        exp_dict = expected.to_dict()
        cand_dict = candidate.to_dict()

        for key in exp_dict:
            if exp_dict[key] != cand_dict[key]:
                mismatches[key] = {
                    "expected": exp_dict[key],
                    "got": cand_dict[key],
                }

        if mismatches:
            return VerificationResult(
                passed=False,
                reason=f"Tile attributes do not match: {list(mismatches.keys())}",
                details={"mismatches": mismatches},
            )

        return VerificationResult(
            passed=True,
            reason="All tile attributes match",
        )

    def difficulty_axes(self) -> list[DifficultyRange]:
        """Declare difficulty parameters for Raven's matrices."""
        return [
            DifficultyRange(
                name="rules",
                min_val=1,
                max_val=4,
                step=1,
                description="Number of transformation rules applied to the matrix",
            ),
            DifficultyRange(
                name="complexity",
                min_val=0,
                max_val=1,
                step=1,
                description="0 = additive (linear), 1 = compositional (row+col)",
            ),
        ]
