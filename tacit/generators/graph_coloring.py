"""Planar Graph k-Coloring generator for TACIT Benchmark.

Generates planar graphs that must be colored with exactly k colors such that
no two adjacent nodes share the same color. Uses Delaunay triangulation with
edge removal for planar graph generation and backtracking for exact k-coloring.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from tacit.core.renderer import (
    STYLE,
    create_canvas,
    draw_circle,
    draw_line,
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

# --- Internal data types ---

class _PuzzleData:
    """Internal representation of a graph coloring puzzle."""

    __slots__ = ("graph", "positions", "node_ids", "k", "edges")

    def __init__(
        self,
        graph: nx.Graph,
        positions: dict[int, tuple[float, float]],
        node_ids: list[int],
        k: int,
        edges: list[tuple[int, int]],
    ) -> None:
        self.graph = graph
        self.positions = positions
        self.node_ids = node_ids
        self.k = k
        self.edges = edges


class _SolutionData:
    """Internal representation of a graph coloring solution."""

    __slots__ = ("coloring",)

    def __init__(self, coloring: dict[int, int]) -> None:
        self.coloring = coloring


# --- Planar graph generation ---


def _generate_planar_graph(
    n_nodes: int,
    edge_density: float,
    rng: np.random.Generator,
) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    """Generate a planar graph using Delaunay triangulation with edge removal.

    Args:
        n_nodes: Number of nodes in the graph.
        edge_density: Fraction of Delaunay edges to keep (0.0 to 1.0).
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple of (graph, positions) where positions maps node id to (x, y).
    """
    # Generate random 2D points
    points = rng.uniform(0.1, 0.9, size=(n_nodes, 2))

    # Build Delaunay triangulation (guaranteed planar)
    tri = Delaunay(points)

    # Create graph from triangulation edges
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                edge = (min(a, b), max(a, b))
                edge_set.add(edge)

    # Remove edges randomly to achieve target density
    all_edges = list(edge_set)
    rng.shuffle(all_edges)

    # Keep at least n_nodes - 1 edges to maintain connectivity
    n_keep = max(n_nodes - 1, int(len(all_edges) * edge_density))
    n_keep = min(n_keep, len(all_edges))

    G.add_edges_from(all_edges[:n_keep])

    # Ensure connectivity: add edges back if needed
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            # Find closest pair between components
            comp_a = list(components[0])
            comp_b = list(components[i])
            best_dist = float("inf")
            best_edge = (comp_a[0], comp_b[0])
            for a in comp_a:
                for b in comp_b:
                    dx = points[a][0] - points[b][0]
                    dy = points[a][1] - points[b][1]
                    dist = dx * dx + dy * dy
                    if dist < best_dist:
                        best_dist = dist
                        best_edge = (a, b)
            G.add_edge(*best_edge)
            # Merge component b into component 0
            components[0] = components[0] | components[i]

    positions = {i: (float(points[i][0]), float(points[i][1])) for i in range(n_nodes)}
    return G, positions


# --- k-coloring via backtracking ---


def _backtrack_coloring(
    graph: nx.Graph,
    k: int,
    node_order: list[int],
    coloring: dict[int, int],
    idx: int,
) -> bool:
    """Backtracking k-coloring solver.

    Args:
        graph: The graph to color.
        k: Number of colors available.
        node_order: Order in which to assign colors to nodes.
        coloring: Current partial coloring (modified in place).
        idx: Current index in node_order.

    Returns:
        True if a valid coloring was found, False otherwise.
    """
    if idx == len(node_order):
        return True

    node = node_order[idx]
    neighbor_colors = {coloring[n] for n in graph.neighbors(node) if n in coloring}

    for color in range(k):
        if color not in neighbor_colors:
            coloring[node] = color
            if _backtrack_coloring(graph, k, node_order, coloring, idx + 1):
                return True
            del coloring[node]

    return False


def _find_k_coloring(
    graph: nx.Graph,
    k: int,
    rng: np.random.Generator,
) -> dict[int, int] | None:
    """Find a valid k-coloring of the graph using backtracking.

    Uses a degree-based ordering (largest first) for efficiency,
    with randomization to break ties for diversity.

    Returns:
        Dict mapping node -> color index (0 to k-1), or None if impossible.
    """
    nodes = list(graph.nodes())
    # Order by degree descending, with random tiebreaking
    degrees = {n: graph.degree(n) for n in nodes}
    noise = {n: float(rng.uniform(0, 0.1)) for n in nodes}
    node_order = sorted(nodes, key=lambda n: (-degrees[n], noise[n]))

    coloring: dict[int, int] = {}
    if _backtrack_coloring(graph, k, node_order, coloring, 0):
        return coloring
    return None


def _ensure_exact_k_colors(
    graph: nx.Graph,
    coloring: dict[int, int],
    k: int,
    rng: np.random.Generator,
) -> dict[int, int]:
    """Adjust coloring to use exactly k colors.

    If the greedy/backtracking solution uses fewer than k colors,
    reassign some nodes to unused colors while maintaining validity.
    """
    used = set(coloring.values())
    if len(used) == k:
        return coloring

    # Need to introduce missing colors
    missing = [c for c in range(k) if c not in used]

    result = dict(coloring)
    nodes = list(graph.nodes())
    rng.shuffle(nodes)

    for color_to_add in missing:
        # Find a node that can safely be recolored to color_to_add
        for node in nodes:
            neighbor_colors = {result[n] for n in graph.neighbors(node)}
            if color_to_add not in neighbor_colors:
                # Check that removing this node's current color won't leave
                # that color unused (unless it's already being removed intentionally)
                old_color = result[node]
                # Count how many other nodes use old_color
                count_old = sum(1 for n in result if n != node and result[n] == old_color)
                if count_old > 0:
                    result[node] = color_to_add
                    break

    return result


# --- SVG rendering ---

_CANVAS_SIZE = 500
_NODE_RADIUS = 16
_MARGIN = 50


def _scale_positions(
    positions: dict[int, tuple[float, float]],
) -> dict[int, tuple[float, float]]:
    """Scale node positions from [0,1] to canvas coordinates with margins."""
    usable = _CANVAS_SIZE - 2 * _MARGIN
    return {
        node: (_MARGIN + x * usable, _MARGIN + y * usable)
        for node, (x, y) in positions.items()
    }


def _render_graph_svg(
    puzzle_data: _PuzzleData,
    coloring: dict[int, int] | None = None,
) -> str:
    """Render the graph to SVG.

    Args:
        puzzle_data: The graph puzzle data.
        coloring: Optional dict mapping node -> color index.
            If None, all nodes are drawn in gray (puzzle view).
            If provided, nodes are drawn in their assigned colors (solution view).

    Returns:
        SVG string.
    """
    canvas = create_canvas(_CANVAS_SIZE, _CANVAS_SIZE)
    scaled = _scale_positions(puzzle_data.positions)
    palette = STYLE["colors"]

    # Draw edges first (behind nodes)
    for u, v in puzzle_data.edges:
        x1, y1 = scaled[u]
        x2, y2 = scaled[v]
        draw_line(canvas, x1, y1, x2, y2, stroke="#999999", stroke_width=1.5)

    # Draw nodes with id attributes for reliable SVG parsing
    for node in puzzle_data.node_ids:
        cx, cy = scaled[node]
        if coloring is not None and node in coloring:
            fill = palette[coloring[node] % len(palette)]
        else:
            fill = "#DDDDDD"
        # Add circle with data-node-id encoded in the id attribute
        circle = canvas.circle(
            center=(cx, cy),
            r=_NODE_RADIUS,
            fill=fill,
            stroke="#333333",
            stroke_width=1.5,
        )
        circle["id"] = f"node-{node}"
        canvas.add(circle)
        # Draw node label
        draw_text(canvas, cx, cy + 5, str(node), font_size=11, fill="#000000")

    return svg_to_string(canvas)


# --- Generator class ---


class GraphColoringGenerator(BaseGenerator):
    """Generates planar graph k-coloring puzzles.

    Creates planar graphs via Delaunay triangulation with edge removal,
    then finds a valid k-coloring using backtracking. The puzzle shows
    the uncolored graph; the solution shows nodes colored with k colors
    such that no adjacent nodes share a color.
    """

    def __init__(self) -> None:
        super().__init__(task_name="graph_coloring")

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[_PuzzleData, _SolutionData]:
        """Generate a planar graph and find a valid k-coloring."""
        n_nodes = int(difficulty.params.get("nodes", 6))
        edge_density = float(difficulty.params.get("edge_density", 0.3))
        k = int(difficulty.params.get("k", 4))

        # Clamp parameters
        n_nodes = max(4, n_nodes)
        k = max(2, min(k, len(STYLE["colors"])))
        edge_density = max(0.1, min(1.0, edge_density))

        # Generate planar graph
        graph, positions = _generate_planar_graph(n_nodes, edge_density, rng)

        # Find k-coloring. If k is too small, increase it.
        coloring = None
        attempt_k = k
        while coloring is None and attempt_k <= len(STYLE["colors"]):
            coloring = _find_k_coloring(graph, attempt_k, rng)
            if coloring is None:
                attempt_k += 1

        if coloring is None:
            # Fallback: use networkx greedy coloring
            nx_coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
            coloring = nx_coloring
            attempt_k = max(coloring.values()) + 1 if coloring else k

        # Actual k used
        k = attempt_k

        # Ensure exactly k colors are used
        coloring = _ensure_exact_k_colors(graph, coloring, k, rng)

        node_ids = sorted(graph.nodes())
        edges = sorted(graph.edges())

        puzzle_data = _PuzzleData(
            graph=graph,
            positions=positions,
            node_ids=node_ids,
            k=k,
            edges=edges,
        )
        solution_data = _SolutionData(coloring=coloring)

        return puzzle_data, solution_data

    def _generate_puzzle_svg(self, puzzle_data: Any) -> str:
        """Render the uncolored graph."""
        return _render_graph_svg(puzzle_data, coloring=None)

    def _generate_solution_svg(
        self, puzzle_data: Any, solution_data: Any
    ) -> str:
        """Render the graph with k-coloring applied."""
        return _render_graph_svg(puzzle_data, coloring=solution_data.coloring)

    def _generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a single distractor with a specific violation.

        Violation types:
            adjacent_conflict: Two adjacent nodes share the same color.
            missing_color: Uses fewer than k colors (k-1 colors).
            wrong_k: Uses more than k colors (k+1 colors).
        """
        pd: _PuzzleData = puzzle_data
        sd: _SolutionData = solution_data
        coloring = dict(sd.coloring)

        if violation_type == "adjacent_conflict":
            coloring = self._make_adjacent_conflict(pd, coloring, rng)
        elif violation_type == "missing_color":
            coloring = self._make_missing_color(pd, coloring, rng)
        elif violation_type == "wrong_k":
            coloring = self._make_wrong_k(pd, coloring, rng)
        else:
            # Default to adjacent_conflict
            coloring = self._make_adjacent_conflict(pd, coloring, rng)
            violation_type = "adjacent_conflict"

        svg = _render_graph_svg(pd, coloring=coloring)
        return svg, violation_type

    def _make_adjacent_conflict(
        self,
        puzzle_data: _PuzzleData,
        coloring: dict[int, int],
        rng: np.random.Generator,
    ) -> dict[int, int]:
        """Create a coloring with at least one adjacent pair sharing a color."""
        result = dict(coloring)
        edges = list(puzzle_data.edges)
        rng.shuffle(edges)

        # Pick a random edge and set both endpoints to the same color
        for u, v in edges:
            result[v] = result[u]
            return result

        return result

    def _make_missing_color(
        self,
        puzzle_data: _PuzzleData,
        coloring: dict[int, int],
        rng: np.random.Generator,
    ) -> dict[int, int]:
        """Create a coloring that uses fewer than k colors."""
        result = dict(coloring)
        k = puzzle_data.k

        if k <= 2:
            # Can't reduce below 1 meaningfully, fall back to adjacent conflict
            return self._make_adjacent_conflict(puzzle_data, coloring, rng)

        # Find a color to eliminate and replace with another
        used_colors = list(set(result.values()))
        if len(used_colors) <= 1:
            return result

        # Pick a color to remove
        color_to_remove = int(rng.choice(used_colors))
        remaining = [c for c in used_colors if c != color_to_remove]
        replacement = int(rng.choice(remaining))

        for node in result:
            if result[node] == color_to_remove:
                result[node] = replacement

        return result

    def _make_wrong_k(
        self,
        puzzle_data: _PuzzleData,
        coloring: dict[int, int],
        rng: np.random.Generator,
    ) -> dict[int, int]:
        """Create a coloring that uses k+1 colors."""
        result = dict(coloring)
        k = puzzle_data.k
        extra_color = k  # 0-indexed, so k is the (k+1)-th color

        # Assign extra color to a random node
        nodes = list(result.keys())
        node = int(rng.choice(nodes))
        result[node] = extra_color

        return result

    def _available_violations(self) -> list[str]:
        """List distractor violation types."""
        return ["adjacent_conflict", "missing_color", "wrong_k"]

    def verify(
        self, puzzle: PuzzleInstance, candidate_png: bytes
    ) -> VerificationResult:
        """Verify candidate PNG has valid k-coloring by sampling node fill colors."""
        from tacit.core.cv_utils import (
            find_closest_palette_color,
            png_to_numpy,
            sample_color,
        )

        # Regenerate puzzle data to get graph structure and node positions
        rng = np.random.default_rng(puzzle.seed)
        puzzle_data, _ = self._generate_puzzle(puzzle.difficulty, rng)

        graph = puzzle_data.graph
        positions = puzzle_data.positions
        k = puzzle_data.k
        nodes = puzzle_data.node_ids
        edges = puzzle_data.edges

        # Scale positions to canvas coordinates (same as rendering)
        scaled = _scale_positions(positions)

        # Build color palette: hex -> index.
        # Include non-fill colors (stroke, edges, background, uncolored)
        # as reject entries (-1) so they never match a fill color.
        palette: dict[str, int] = {}
        for i in range(len(STYLE["colors"])):
            palette[STYLE["colors"][i].upper()] = i
        for reject_hex in ("#DDDDDD", "#333333", "#999999", "#000000", "#FFFFFF"):
            palette[reject_hex] = -1

        img = png_to_numpy(candidate_png)

        # Nodes are drawn in node_ids order (sorted ascending), so
        # higher-index nodes paint over lower-index ones.  We need
        # occlusion-aware sampling to avoid reading the wrong color.
        _R = _NODE_RADIUS

        def _is_occluded(
            sx: float, sy: float, later: list[int],
        ) -> bool:
            """Check if (sx, sy) falls inside a later-drawn node's circle."""
            for ln in later:
                lx, ly = scaled[ln]
                if (sx - lx) ** 2 + (sy - ly) ** 2 < _R * _R:
                    return True
            return False

        def _sample_node_color(
            node: int,
        ) -> int | None:
            """Sample the fill color of *node* from the rasterized image.

            Uses a multi-strategy approach:
            1. Try fixed offsets that avoid the text label.
            2. If all fixed offsets are occluded, generate directed
               sample points away from occluding nodes.
            """
            cx, cy = scaled[node]

            # Nodes drawn later (higher index) may occlude this one.
            later = [
                n for n in nodes
                if n > node
                and (scaled[n][0] - cx) ** 2 + (scaled[n][1] - cy) ** 2
                < (3 * _R) ** 2
            ]

            # Strategy 1: fixed candidate offsets
            offsets = [
                (5, -_R // 2), (-5, -_R // 2),
                (5, _R // 2 + 3), (-5, _R // 2 + 3),
                (_R // 2, 0), (-_R // 2, 0),
                (0, -_R // 2), (_R // 2, -3), (-_R // 2, -3),
            ]

            # Strategy 2: if we have occluding neighbours, add directed
            # sample points opposite to each occluder (within circle,
            # avoiding text area y ∈ [cy-2, cy+6]).
            for ln in later:
                lx, ly = scaled[ln]
                dx, dy = cx - lx, cy - ly
                length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
                # Point at ~70% radius away from the occluder
                r_sample = _R * 0.7
                ox = int(round(dx / length * r_sample))
                oy = int(round(dy / length * r_sample))
                # Avoid the text band at center
                if -2 <= oy <= 6:
                    oy = -3 if oy < 2 else 7
                offsets.append((ox, oy))

            votes: dict[int, int] = {}
            for dx, dy in offsets:
                sx, sy = cx + dx, cy + dy
                if _is_occluded(sx, sy, later):
                    continue
                pixel = sample_color(img, int(sx), int(sy))
                idx = find_closest_palette_color(pixel, palette, threshold=50)
                if idx is not None and idx >= 0:
                    votes[idx] = votes.get(idx, 0) + 1

            if not votes:
                return None
            return max(votes, key=lambda v: votes[v])

        node_colors: dict[int, int] = {}
        for node in nodes:
            color_idx = _sample_node_color(node)
            if color_idx is None:
                cx, cy = scaled[node]
                return VerificationResult(
                    passed=False,
                    reason=f"Node {node} has unrecognized color at ({cx:.0f}, {cy:.0f}).",
                )
            node_colors[node] = color_idx

        # Check all nodes colored
        if len(node_colors) != len(nodes):
            return VerificationResult(
                passed=False,
                reason=f"Only {len(node_colors)}/{len(nodes)} nodes detected.",
            )

        # Check adjacency constraint
        conflicts = 0
        for u, v in edges:
            if node_colors.get(u) == node_colors.get(v):
                conflicts += 1

        if conflicts > 0:
            return VerificationResult(
                passed=False,
                reason=f"{conflicts} adjacent node pair(s) share a color.",
                details={"adjacent_conflicts": conflicts},
            )

        # Check exactly k colors used
        unique_colors = set(node_colors.values())
        colors_used = len(unique_colors)
        if colors_used != k:
            return VerificationResult(
                passed=False,
                reason=f"Expected {k} colors, found {colors_used}.",
                details={"colors_used": colors_used},
            )

        return VerificationResult(
            passed=True,
            details={"adjacent_conflicts": 0, "colors_used": colors_used},
        )

    def difficulty_axes(self) -> list[DifficultyRange]:
        """Declare the difficulty parameters for graph coloring."""
        return [
            DifficultyRange(
                name="nodes",
                min_val=4,
                max_val=50,
                step=1,
                description="Number of nodes in the graph",
            ),
            DifficultyRange(
                name="edge_density",
                min_val=0.1,
                max_val=1.0,
                step=0.1,
                description="Fraction of possible planar edges to keep",
            ),
            DifficultyRange(
                name="k",
                min_val=2,
                max_val=10,
                step=1,
                description="Number of colors for the coloring",
            ),
        ]
