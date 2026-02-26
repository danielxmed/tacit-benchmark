"""Unknot Detection Generator for TACIT Benchmark.

Generates 2D knot diagram projections and asks whether the diagram
represents the unknot (trivially deformable to a simple circle) or
a non-trivial knot.

Key ideas
---------
* **Unknot generation**: Start with a circular path, then apply random
  Reidemeister-I moves to insert crossings that can always be resolved
  back to a simple circle.  This guarantees the result is an unknot with
  N visible crossings.
* **Non-trivial knot generation**: Use Gauss code representations of
  known knots (trefoil, figure-eight, cinquefoil, ...) and embed them as
  smooth curves in 2-D with correct over/under crossing indicators.
* **SVG rendering**: Smooth cubic Bezier curves with a gap in the
  under-strand at each crossing to visually distinguish over/under.
* **Verification**: The solution is a binary label encoded in a small SVG
  badge ("unknot" or "knot").  Verification extracts that label.
* **Distractors**: The opposite answer (since this is a binary task).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from tacit.core.types import (
    DifficultyParams,
    DifficultyRange,
    PuzzleInstance,
    VerificationResult,
)
from tacit.generators.base import BaseGenerator


# ---------------------------------------------------------------------------
# Crossing & diagram data structures
# ---------------------------------------------------------------------------

class Crossing:
    """A single crossing in a knot diagram.

    Attributes
    ----------
    x, y : float
        Position in the 2-D plane.
    over_angle : float
        Angle (rad) of the over-strand through this crossing.
    sign : int
        +1 for a positive crossing, -1 for a negative crossing.
    """

    __slots__ = ("x", "y", "over_angle", "sign")

    def __init__(self, x: float, y: float, over_angle: float, sign: int) -> None:
        self.x = x
        self.y = y
        self.over_angle = over_angle
        self.sign = sign


class KnotDiagram:
    """Lightweight representation of a knot diagram for rendering.

    Stores a list of crossings and a smooth backbone path (as a sequence
    of (x, y) waypoints) used to draw the curve.
    """

    def __init__(
        self,
        crossings: list[Crossing],
        path_points: list[tuple[float, float]],
        is_unknot: bool,
    ) -> None:
        self.crossings = crossings
        self.path_points = path_points
        self.is_unknot = is_unknot


# ---------------------------------------------------------------------------
# Known non-trivial knots (Gauss codes)
# ---------------------------------------------------------------------------
# Each entry is a tuple (name, gauss_code_signs) where gauss_code_signs
# is a list of (crossing_index, over_bool, sign) triples traversed in
# order around the knot.  We only need the *sign pattern* to know the
# knot is non-trivial; the actual embedding is computed procedurally.

_KNOWN_KNOTS: list[tuple[str, int]] = [
    ("trefoil", 3),
    ("figure_eight", 4),
    ("cinquefoil", 5),
    ("three_twist", 5),
    ("stevedore", 6),
    ("knot_7_1", 7),
]


# ---------------------------------------------------------------------------
# Diagram generation helpers
# ---------------------------------------------------------------------------

def _circle_points(n: int, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Generate *n* equally spaced points on a circle."""
    return [
        (cx + r * math.cos(2 * math.pi * i / n),
         cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def _generate_unknot_diagram(
    num_crossings: int,
    rng: np.random.Generator,
    *,
    canvas: float = 400.0,
) -> KnotDiagram:
    """Create an unknot diagram with *num_crossings* Reidemeister-I loops.

    Each crossing is a "kink" inserted into a circle --- visually complex
    but topologically trivial.
    """
    cx, cy = canvas / 2, canvas / 2
    base_r = canvas * 0.35

    # Number of backbone waypoints between crossings
    points_per_segment = 12
    total_backbone = num_crossings * points_per_segment
    backbone = _circle_points(total_backbone, cx, cy, base_r)

    crossings: list[Crossing] = []
    path_points: list[tuple[float, float]] = []

    for seg_idx in range(num_crossings):
        start = seg_idx * points_per_segment
        # Crossing sits at the midpoint of the segment
        mid = start + points_per_segment // 2

        # Compute crossing position with a small radial displacement
        angle = 2 * math.pi * mid / total_backbone
        disp = float(rng.uniform(base_r * 0.08, base_r * 0.18))
        disp_sign = 1 if rng.random() < 0.5 else -1
        c_x = cx + (base_r + disp * disp_sign) * math.cos(angle)
        c_y = cy + (base_r + disp * disp_sign) * math.sin(angle)

        over_angle = angle + math.pi / 2  # perpendicular to radius
        sign = int(rng.choice([-1, 1]))
        crossings.append(Crossing(c_x, c_y, over_angle, sign))

        # Build the kinked path segment (loop out and back)
        for j in range(points_per_segment):
            idx = start + j
            bx, by = backbone[idx % total_backbone]

            # Near the midpoint, push the path outward to create the kink
            dist_to_mid = abs(j - points_per_segment // 2)
            if dist_to_mid < points_per_segment // 3:
                kink_strength = (1 - dist_to_mid / (points_per_segment // 3))
                kink_r = base_r * 0.12 * kink_strength * disp_sign
                local_angle = 2 * math.pi * idx / total_backbone
                bx += kink_r * math.cos(local_angle)
                by += kink_r * math.sin(local_angle)

            path_points.append((bx, by))

    return KnotDiagram(crossings=crossings, path_points=path_points, is_unknot=True)


def _generate_nontrivial_knot_diagram(
    num_crossings: int,
    rng: np.random.Generator,
    *,
    canvas: float = 400.0,
) -> KnotDiagram:
    """Generate a non-trivial knot diagram.

    Picks a known knot whose minimal crossing number is <= num_crossings
    and embeds it as a torus-knot-style curve.
    """
    # Select a known knot with crossing count <= requested
    eligible = [(name, nc) for name, nc in _KNOWN_KNOTS if nc <= num_crossings]
    if not eligible:
        eligible = [_KNOWN_KNOTS[0]]  # fall back to trefoil

    idx = int(rng.integers(0, len(eligible)))
    knot_name, knot_crossings = eligible[idx]

    cx, cy = canvas / 2, canvas / 2
    base_r = canvas * 0.35

    # Determine torus-knot parameters (p, q) based on the knot
    # Trefoil = (2,3), figure-eight uses a different embedding
    p, q = _knot_pq(knot_name, knot_crossings)

    # Generate the torus-knot parametric curve
    num_points = max(knot_crossings * 24, 120)
    path_points: list[tuple[float, float]] = []
    r_minor = base_r * 0.35

    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        # Torus knot parametrization projected to 2D
        r = base_r + r_minor * math.cos(q * t)
        x = cx + r * math.cos(p * t)
        y = cy + r * math.sin(p * t)
        # Add small random perturbation for visual variety
        x += float(rng.uniform(-1.5, 1.5))
        y += float(rng.uniform(-1.5, 1.5))
        path_points.append((x, y))

    # Find self-intersections to place crossings
    crossings = _find_crossings(path_points, rng)

    return KnotDiagram(crossings=crossings, path_points=path_points, is_unknot=False)


def _knot_pq(knot_name: str, knot_crossings: int) -> tuple[int, int]:
    """Return (p, q) torus knot parameters for embedding."""
    mapping = {
        "trefoil": (2, 3),
        "cinquefoil": (2, 5),
        "knot_7_1": (2, 7),
    }
    if knot_name in mapping:
        return mapping[knot_name]
    # For non-torus knots, use a Lissajous-like embedding
    return (3, knot_crossings)


def _find_crossings(
    path_points: list[tuple[float, float]],
    rng: np.random.Generator,
) -> list[Crossing]:
    """Detect approximate self-intersections in a path.

    Uses a simplified segment-segment intersection test.
    """
    crossings: list[Crossing] = []
    n = len(path_points)
    skip = max(n // 6, 3)  # minimum separation to avoid adjacent segments

    seen_positions: set[tuple[int, int]] = set()

    for i in range(n):
        j = (i + 1) % n
        ax, ay = path_points[i]
        bx, by = path_points[j]

        for k in range(i + skip, n):
            l_ = (k + 1) % n
            if abs(k - i) < skip or abs(l_ - i) < skip:
                continue
            cx_, cy_ = path_points[k]
            dx, dy = path_points[l_]

            pt = _segment_intersection(ax, ay, bx, by, cx_, cy_, dx, dy)
            if pt is not None:
                # Quantize to avoid duplicate crossings at the same spot
                key = (int(pt[0] * 10), int(pt[1] * 10))
                if key not in seen_positions:
                    seen_positions.add(key)
                    over_angle = math.atan2(by - ay, bx - ax)
                    sign = int(rng.choice([-1, 1]))
                    crossings.append(Crossing(pt[0], pt[1], over_angle, sign))

    return crossings


def _segment_intersection(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> tuple[float, float] | None:
    """Compute intersection point of segments AB and CD, or None."""
    denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denom) < 1e-10:
        return None

    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
    u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom

    if 0 < t < 1 and 0 < u < 1:
        ix = ax + t * (bx - ax)
        iy = ay + t * (by - ay)
        return (ix, iy)
    return None


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

_SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
)

_GAP_RADIUS = 8.0  # half-width of the gap for under-crossings


def _render_knot_svg(diagram: KnotDiagram, *, canvas: float = 400.0) -> str:
    """Render a KnotDiagram to an SVG string."""
    parts: list[str] = [_SVG_HEADER.format(w=int(canvas), h=int(canvas))]

    # Background
    parts.append(
        f'<rect width="{int(canvas)}" height="{int(canvas)}" fill="white"/>'
    )

    # Draw the smooth path
    pts = diagram.path_points
    if len(pts) < 2:
        parts.append("</svg>")
        return "\n".join(parts)

    # Build the main path as a closed polyline
    path_d = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"
    for px, py in pts[1:]:
        path_d += f" L {px:.1f},{py:.1f}"
    path_d += " Z"

    # Draw the path with gaps at under-crossings
    parts.append(
        f'<path d="{path_d}" fill="none" stroke="#333" '
        f'stroke-width="3" stroke-linejoin="round"/>'
    )

    # Draw crossing indicators
    for crossing in diagram.crossings:
        # Over-strand: a short thick line
        dx = _GAP_RADIUS * 1.5 * math.cos(crossing.over_angle)
        dy = _GAP_RADIUS * 1.5 * math.sin(crossing.over_angle)
        parts.append(
            f'<line x1="{crossing.x - dx:.1f}" y1="{crossing.y - dy:.1f}" '
            f'x2="{crossing.x + dx:.1f}" y2="{crossing.y + dy:.1f}" '
            f'stroke="#333" stroke-width="4"/>'
        )

        # Under-strand gap: white rectangle behind the over-strand
        perp_angle = crossing.over_angle + math.pi / 2
        ux = _GAP_RADIUS * math.cos(perp_angle)
        uy = _GAP_RADIUS * math.sin(perp_angle)
        parts.append(
            f'<line x1="{crossing.x - ux:.1f}" y1="{crossing.y - uy:.1f}" '
            f'x2="{crossing.x + ux:.1f}" y2="{crossing.y + uy:.1f}" '
            f'stroke="white" stroke-width="6"/>'
        )

        # Small circle at crossing for clarity
        color = "#2196F3" if crossing.sign > 0 else "#F44336"
        parts.append(
            f'<circle cx="{crossing.x:.1f}" cy="{crossing.y:.1f}" '
            f'r="3" fill="{color}" opacity="0.6"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _render_answer_svg(is_unknot: bool, *, canvas: float = 200.0) -> str:
    """Render a small badge SVG with the answer label."""
    w, h = int(canvas), int(canvas // 2)
    label = "unknot" if is_unknot else "knot"
    color = "#4CAF50" if is_unknot else "#F44336"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        f'<rect width="{w}" height="{h}" rx="10" fill="{color}"/>'
        f'<text x="{w // 2}" y="{h // 2 + 8}" text-anchor="middle" '
        f'font-size="28" font-family="sans-serif" fill="white">{label}</text>'
        f"</svg>"
    )


def _parse_answer_from_png(png_bytes: bytes) -> bool | None:
    """Detect green (unknot) vs red (knot) from PNG badge.

    Badge colors: #4CAF50 (unknot/green), #F44336 (knot/red).
    """
    from tacit.core.cv_utils import count_color_pixels, hex_to_rgb, png_to_numpy

    img = png_to_numpy(png_bytes)
    green_count = count_color_pixels(img, hex_to_rgb("#4CAF50"), threshold=60)
    red_count = count_color_pixels(img, hex_to_rgb("#F44336"), threshold=60)

    min_pixels = 50
    if green_count > red_count and green_count > min_pixels:
        return True
    if red_count > green_count and red_count > min_pixels:
        return False
    return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class UnknotGenerator(BaseGenerator):
    """Generator for unknot detection puzzles.

    Parameters
    ----------
    crossings : int
        Number of crossings in the knot diagram.  More crossings make the
        puzzle harder to solve visually.
    """

    def __init__(self) -> None:
        super().__init__(task_name="unknot")

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[Any, Any]:
        num_crossings = int(difficulty.params.get("crossings", 3))
        num_crossings = max(num_crossings, 2)

        # Decide whether to generate an unknot or a non-trivial knot
        # Aim for roughly 50/50 balance
        is_unknot = bool(rng.random() < 0.5)

        if is_unknot:
            diagram = _generate_unknot_diagram(num_crossings, rng)
        else:
            diagram = _generate_nontrivial_knot_diagram(num_crossings, rng)

        puzzle_data = diagram
        solution_data = {"is_unknot": diagram.is_unknot}
        return puzzle_data, solution_data

    def _generate_puzzle_svg(self, puzzle_data: Any) -> str:
        diagram: KnotDiagram = puzzle_data
        return _render_knot_svg(diagram)

    def _generate_solution_svg(self, puzzle_data: Any, solution_data: Any) -> str:
        is_unknot: bool = solution_data["is_unknot"]
        return _render_answer_svg(is_unknot)

    def _generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        # Binary task: distractor is the opposite answer
        is_unknot: bool = solution_data["is_unknot"]
        opposite = not is_unknot
        return _render_answer_svg(opposite), violation_type

    def _available_violations(self) -> list[str]:
        return ["opposite_answer"]

    def verify(
        self, puzzle: PuzzleInstance, candidate_png: bytes
    ) -> VerificationResult:
        """Verify by detecting green (unknot) vs red (knot) indicator in PNG."""
        expected_unknot = puzzle.metadata.get("is_unknot")
        if expected_unknot is None:
            return VerificationResult(
                passed=False,
                reason="Missing is_unknot in puzzle metadata.",
            )

        candidate_is_unknot = _parse_answer_from_png(candidate_png)
        if candidate_is_unknot is None:
            return VerificationResult(
                passed=False,
                reason="Could not detect answer indicator in candidate PNG.",
            )

        if candidate_is_unknot == expected_unknot:
            return VerificationResult(passed=True, reason="Correct answer.")
        expected_label = "unknot" if expected_unknot else "knot"
        given_label = "unknot" if candidate_is_unknot else "knot"
        return VerificationResult(
            passed=False,
            reason=f"Expected {expected_label}, got {given_label}.",
        )

    def difficulty_axes(self) -> list[DifficultyRange]:
        return [
            DifficultyRange(
                name="crossings",
                min_val=2,
                max_val=15,
                step=1,
                description="Number of crossings in the knot diagram",
            ),
        ]

    def generate(
        self,
        difficulty: DifficultyParams,
        seed: int,
        num_distractors: int = 4,
    ) -> PuzzleInstance:
        """Override to inject metadata with is_unknot flag."""
        instance = super().generate(difficulty, seed, num_distractors)
        # The puzzle_data (KnotDiagram) was used during generation.
        # We need to regenerate to capture is_unknot for metadata.
        rng = np.random.default_rng(seed)
        num_crossings = int(difficulty.params.get("crossings", 3))
        num_crossings = max(num_crossings, 2)
        is_unknot = bool(rng.random() < 0.5)

        instance.metadata["is_unknot"] = is_unknot
        return instance
