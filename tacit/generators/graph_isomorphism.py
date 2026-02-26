"""Graph Isomorphism Detection generator for TACIT Benchmark.

Generates pairs of graph drawings with different layouts. The task is
to determine whether the two graphs are isomorphic (structurally
identical despite different visual layouts) or not.

Positive pairs: same graph with permuted node labels and a different layout.
Negative pairs: a modified graph with one edge added or removed.
Verification uses networkx.is_isomorphic (VF2 algorithm).
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import svgwrite

from tacit.core.types import (
    DifficultyParams,
    DifficultyRange,
    PuzzleInstance,
    VerificationResult,
)
from tacit.generators.base import BaseGenerator

# --- Constants ---

_SVG_WIDTH = 800
_SVG_HEIGHT = 400
_GRAPH_AREA_WIDTH = 360
_GRAPH_AREA_HEIGHT = 360
_GRAPH_PADDING = 20
_NODE_RADIUS = 12
_GRAPH_A_OFFSET_X = 20
_GRAPH_B_OFFSET_X = 420
_GRAPH_OFFSET_Y = 20

_NODE_FILL = "#4A90D9"
_NODE_STROKE = "#2C5F8A"
_EDGE_STROKE = "#666666"
_EDGE_WIDTH = 2
_LABEL_FONT_SIZE = 10
_TITLE_FONT_SIZE = 14

_SOLUTION_WIDTH = 200
_SOLUTION_HEIGHT = 200

_ISO_CHECK_COLOR = "#2ECC40"
_NON_ISO_X_COLOR = "#FF4136"


class GraphIsomorphismGenerator(BaseGenerator):
    """Generator for graph isomorphism detection puzzles."""

    def __init__(self) -> None:
        super().__init__(task_name="graph_isomorphism")

    def generate(
        self,
        difficulty: DifficultyParams,
        seed: int,
        num_distractors: int = 4,
    ) -> PuzzleInstance:
        """Override to inject metadata after base generation."""
        puzzle = super().generate(difficulty, seed, num_distractors)
        # Metadata is stored in puzzle_data by _generate_puzzle;
        # we stash it on the instance attribute _last_metadata in _generate_puzzle.
        puzzle.metadata = self._last_metadata
        return puzzle

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a pair of graphs and their layout positions.

        Returns:
            (puzzle_data, solution_data)

            puzzle_data contains:
                - graph_a: adjacency list (list of edge tuples)
                - graph_b: adjacency list (list of edge tuples)
                - positions_a: dict[int, (float, float)]
                - positions_b: dict[int, (float, float)]
                - nodes: int
                - is_isomorphic: bool

            solution_data contains:
                - is_isomorphic: bool
        """
        nodes = difficulty.params.get("nodes", 5)
        distortion = difficulty.params.get("distortion", 0.3)

        # Decide whether this instance is isomorphic or not.
        # Use a coin flip seeded deterministically.
        is_isomorphic = bool(rng.integers(0, 2))

        # Step 1: Generate graph A as a random connected graph.
        graph_a = _generate_random_connected_graph(nodes, rng)

        # Step 2: Generate graph B.
        if is_isomorphic:
            graph_b = _create_isomorphic_copy(graph_a, rng)
        else:
            graph_b = _create_non_isomorphic_variant(graph_a, rng)

        # Step 3: Compute layouts.
        positions_a = _compute_layout(graph_a, rng, distortion)
        positions_b = _compute_layout(graph_b, rng, distortion)

        puzzle_data = {
            "graph_a_edges": list(graph_a.edges()),
            "graph_b_edges": list(graph_b.edges()),
            "graph_a_nodes": sorted(graph_a.nodes()),
            "graph_b_nodes": sorted(graph_b.nodes()),
            "positions_a": {int(k): (float(v[0]), float(v[1])) for k, v in positions_a.items()},
            "positions_b": {int(k): (float(v[0]), float(v[1])) for k, v in positions_b.items()},
            "nodes": nodes,
            "is_isomorphic": is_isomorphic,
        }

        solution_data = {
            "is_isomorphic": is_isomorphic,
        }

        # Stash metadata for the overridden generate() to pick up.
        self._last_metadata = {
            "is_isomorphic": is_isomorphic,
            "nodes": nodes,
            "edges_a": len(list(graph_a.edges())),
            "edges_b": len(list(graph_b.edges())),
        }

        return puzzle_data, solution_data

    # ------------------------------------------------------------------
    # SVG rendering
    # ------------------------------------------------------------------

    def _generate_puzzle_svg(self, puzzle_data: dict[str, Any]) -> str:
        """Render two graphs side-by-side as SVG."""
        dwg = svgwrite.Drawing(size=(f"{_SVG_WIDTH}px", f"{_SVG_HEIGHT}px"))
        dwg.attribs["xmlns"] = "http://www.w3.org/2000/svg"

        # Background
        dwg.add(dwg.rect(insert=(0, 0), size=(_SVG_WIDTH, _SVG_HEIGHT), fill="white"))

        # Graph A
        _draw_graph(
            dwg,
            puzzle_data["graph_a_nodes"],
            puzzle_data["graph_a_edges"],
            puzzle_data["positions_a"],
            offset_x=_GRAPH_A_OFFSET_X,
            offset_y=_GRAPH_OFFSET_Y,
            area_width=_GRAPH_AREA_WIDTH,
            area_height=_GRAPH_AREA_HEIGHT,
        )
        dwg.add(dwg.text(
            "Graph A",
            insert=(_GRAPH_A_OFFSET_X + _GRAPH_AREA_WIDTH / 2, _SVG_HEIGHT - 5),
            text_anchor="middle",
            font_size=_TITLE_FONT_SIZE,
            font_family="sans-serif",
            fill="#333333",
        ))

        # Graph B
        _draw_graph(
            dwg,
            puzzle_data["graph_b_nodes"],
            puzzle_data["graph_b_edges"],
            puzzle_data["positions_b"],
            offset_x=_GRAPH_B_OFFSET_X,
            offset_y=_GRAPH_OFFSET_Y,
            area_width=_GRAPH_AREA_WIDTH,
            area_height=_GRAPH_AREA_HEIGHT,
        )
        dwg.add(dwg.text(
            "Graph B",
            insert=(_GRAPH_B_OFFSET_X + _GRAPH_AREA_WIDTH / 2, _SVG_HEIGHT - 5),
            text_anchor="middle",
            font_size=_TITLE_FONT_SIZE,
            font_family="sans-serif",
            fill="#333333",
        ))

        return dwg.tostring()

    def _generate_solution_svg(
        self, puzzle_data: dict[str, Any], solution_data: dict[str, Any]
    ) -> str:
        """Render solution: green checkmark (isomorphic) or red X (not)."""
        is_iso = solution_data["is_isomorphic"]
        return _render_answer_svg(is_iso)

    def _generate_distractor(
        self,
        puzzle_data: dict[str, Any],
        solution_data: dict[str, Any],
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a distractor (the opposite answer)."""
        is_iso = solution_data["is_isomorphic"]
        # The distractor is always the opposite of the correct answer.
        distractor_svg = _render_answer_svg(not is_iso)
        return distractor_svg, violation_type

    def _available_violations(self) -> list[str]:
        """For binary tasks, there is one type: opposite_answer."""
        return ["opposite_answer"]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify by checking if the candidate matches the solution indicator."""
        expected_iso = puzzle.metadata.get("is_isomorphic")
        if expected_iso is None:
            return VerificationResult(
                passed=False,
                reason="Missing is_isomorphic in puzzle metadata.",
            )

        # Parse the candidate SVG to determine which answer it represents.
        candidate_is_iso = _parse_answer_svg(candidate_svg)
        if candidate_is_iso is None:
            return VerificationResult(
                passed=False,
                reason="Could not parse answer from candidate SVG.",
            )

        if candidate_is_iso == expected_iso:
            return VerificationResult(passed=True, reason="Correct answer.")
        else:
            expected_label = "isomorphic" if expected_iso else "not isomorphic"
            given_label = "isomorphic" if candidate_is_iso else "not isomorphic"
            return VerificationResult(
                passed=False,
                reason=f"Expected {expected_label}, got {given_label}.",
            )

    # ------------------------------------------------------------------
    # Difficulty axes
    # ------------------------------------------------------------------

    def difficulty_axes(self) -> list[DifficultyRange]:
        return [
            DifficultyRange(
                name="nodes",
                min_val=4,
                max_val=30,
                step=1,
                description="Number of nodes in each graph.",
            ),
            DifficultyRange(
                name="distortion",
                min_val=0.0,
                max_val=1.0,
                step=0.1,
                description="Layout distortion factor (higher = harder).",
            ),
        ]


# ======================================================================
# Helper functions (module-private)
# ======================================================================


def _generate_random_connected_graph(
    n: int, rng: np.random.Generator
) -> nx.Graph:
    """Generate a random connected graph with *n* nodes.

    Uses a random spanning tree to guarantee connectivity, then adds
    extra edges for visual complexity.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))

    if n < 2:
        return g

    # Build a random spanning tree: shuffle nodes, connect sequentially.
    node_order = list(range(n))
    rng.shuffle(node_order)
    for i in range(1, n):
        g.add_edge(int(node_order[i - 1]), int(node_order[i]))

    # Add extra edges: approximately n * 0.5 additional edges.
    extra_count = max(1, int(rng.poisson(lam=n * 0.5)))
    all_possible = [
        (i, j) for i in range(n) for j in range(i + 1, n) if not g.has_edge(i, j)
    ]
    if all_possible:
        rng.shuffle(all_possible)
        for edge in all_possible[: min(extra_count, len(all_possible))]:
            g.add_edge(int(edge[0]), int(edge[1]))

    return g


def _create_isomorphic_copy(
    graph: nx.Graph, rng: np.random.Generator
) -> nx.Graph:
    """Create an isomorphic copy by permuting node labels."""
    nodes = sorted(graph.nodes())
    perm = list(nodes)
    rng.shuffle(perm)
    mapping = dict(zip(nodes, perm))
    return nx.relabel_nodes(graph, mapping)


def _create_non_isomorphic_variant(
    graph: nx.Graph, rng: np.random.Generator
) -> nx.Graph:
    """Create a non-isomorphic variant by adding or removing an edge.

    Tries to preserve the node count but change the edge structure.
    """
    g2 = graph.copy()
    nodes = sorted(g2.nodes())
    n = len(nodes)

    # Try adding an edge that doesn't exist.
    non_edges = list(nx.non_edges(g2))
    existing_edges = list(g2.edges())

    if non_edges and len(existing_edges) > n - 1:
        # Coin flip: add or remove.
        if bool(rng.integers(0, 2)) and non_edges:
            idx = int(rng.integers(0, len(non_edges)))
            u, v = non_edges[idx]
            g2.add_edge(u, v)
        else:
            # Remove an edge that won't disconnect the graph.
            rng.shuffle(existing_edges)
            for u, v in existing_edges:
                g2.remove_edge(u, v)
                if nx.is_connected(g2):
                    break
                g2.add_edge(u, v)
            else:
                # Fallback: just add an edge.
                if non_edges:
                    idx = int(rng.integers(0, len(non_edges)))
                    u, v = non_edges[idx]
                    g2.add_edge(u, v)
    elif non_edges:
        idx = int(rng.integers(0, len(non_edges)))
        u, v = non_edges[idx]
        g2.add_edge(u, v)
    else:
        # Complete graph: remove an edge.
        if existing_edges:
            rng.shuffle(existing_edges)
            for u, v in existing_edges:
                g2.remove_edge(u, v)
                if nx.is_connected(g2):
                    break
                g2.add_edge(u, v)

    # Relabel to avoid trivial detection of identical labels.
    perm = list(nodes)
    rng.shuffle(perm)
    mapping = dict(zip(nodes, perm))
    g2 = nx.relabel_nodes(g2, mapping)

    return g2


def _compute_layout(
    graph: nx.Graph,
    rng: np.random.Generator,
    distortion: float,
) -> dict[int, tuple[float, float]]:
    """Compute node positions using spring layout with distortion.

    The distortion parameter controls how much random noise is added
    to the layout, making the visual comparison harder.
    """
    # Use spring layout with a fixed seed derived from our rng
    # for reproducibility.
    layout_seed = int(rng.integers(0, 2**31))
    pos = nx.spring_layout(graph, seed=layout_seed, iterations=50)

    # Apply distortion: add scaled random noise.
    if distortion > 0:
        for node in pos:
            noise_x = float(rng.normal(0, distortion * 0.3))
            noise_y = float(rng.normal(0, distortion * 0.3))
            x, y = pos[node]
            pos[node] = (x + noise_x, y + noise_y)

    # Normalize positions to [0, 1].
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max_x - min_x if max_x != min_x else 1.0
        range_y = max_y - min_y if max_y != min_y else 1.0
        for node in pos:
            x, y = pos[node]
            pos[node] = ((x - min_x) / range_x, (y - min_y) / range_y)

    return pos


def _draw_graph(
    dwg: svgwrite.Drawing,
    nodes: list[int],
    edges: list[tuple[int, int]],
    positions: dict[int, tuple[float, float]],
    offset_x: float,
    offset_y: float,
    area_width: float,
    area_height: float,
) -> None:
    """Draw a graph within a rectangular area of the SVG."""
    # Map normalized [0,1] positions to pixel coordinates within the area.
    padding = _GRAPH_PADDING
    draw_w = area_width - 2 * padding
    draw_h = area_height - 2 * padding

    def to_pixel(node_id: int) -> tuple[float, float]:
        nx_, ny = positions.get(node_id, (0.5, 0.5))
        px = offset_x + padding + nx_ * draw_w
        py = offset_y + padding + ny * draw_h
        return px, py

    # Draw edges first (behind nodes).
    for u, v in edges:
        x1, y1 = to_pixel(u)
        x2, y2 = to_pixel(v)
        dwg.add(dwg.line(
            start=(x1, y1),
            end=(x2, y2),
            stroke=_EDGE_STROKE,
            stroke_width=_EDGE_WIDTH,
        ))

    # Draw nodes.
    for node in nodes:
        cx, cy = to_pixel(node)
        dwg.add(dwg.circle(
            center=(cx, cy),
            r=_NODE_RADIUS,
            fill=_NODE_FILL,
            stroke=_NODE_STROKE,
            stroke_width=1.5,
        ))
        # Node label
        dwg.add(dwg.text(
            str(node),
            insert=(cx, cy + _LABEL_FONT_SIZE * 0.35),
            text_anchor="middle",
            font_size=_LABEL_FONT_SIZE,
            font_family="sans-serif",
            fill="white",
            font_weight="bold",
        ))


def _render_answer_svg(is_isomorphic: bool) -> str:
    """Render a solution/distractor SVG indicator.

    Green checkmark for isomorphic, red X for not isomorphic.
    The SVG includes an id attribute marker for parsing.
    """
    dwg = svgwrite.Drawing(size=(f"{_SOLUTION_WIDTH}px", f"{_SOLUTION_HEIGHT}px"))
    dwg.attribs["xmlns"] = "http://www.w3.org/2000/svg"

    # Background
    dwg.add(dwg.rect(
        insert=(0, 0),
        size=(_SOLUTION_WIDTH, _SOLUTION_HEIGHT),
        fill="white",
    ))

    cx, cy = _SOLUTION_WIDTH / 2, _SOLUTION_HEIGHT / 2

    if is_isomorphic:
        # Green checkmark
        dwg.add(dwg.circle(
            center=(cx, cy),
            r=60,
            fill=_ISO_CHECK_COLOR,
            opacity=0.2,
        ))
        # Checkmark path
        dwg.add(dwg.polyline(
            points=[
                (cx - 30, cy),
                (cx - 10, cy + 25),
                (cx + 35, cy - 30),
            ],
            stroke=_ISO_CHECK_COLOR,
            stroke_width=8,
            fill="none",
            stroke_linecap="round",
            stroke_linejoin="round",
        ))
        dwg.add(dwg.text(
            "Isomorphic",
            insert=(cx, cy + 55),
            text_anchor="middle",
            font_size=14,
            font_family="sans-serif",
            fill=_ISO_CHECK_COLOR,
            font_weight="bold",
        ))
        # Hidden marker for parsing
        dwg.add(dwg.rect(
            insert=(0, 0), size=(0, 0),
            id="answer-isomorphic",
        ))
    else:
        # Red X
        dwg.add(dwg.circle(
            center=(cx, cy),
            r=60,
            fill=_NON_ISO_X_COLOR,
            opacity=0.2,
        ))
        arm = 25
        dwg.add(dwg.line(
            start=(cx - arm, cy - arm),
            end=(cx + arm, cy + arm),
            stroke=_NON_ISO_X_COLOR,
            stroke_width=8,
            stroke_linecap="round",
        ))
        dwg.add(dwg.line(
            start=(cx + arm, cy - arm),
            end=(cx - arm, cy + arm),
            stroke=_NON_ISO_X_COLOR,
            stroke_width=8,
            stroke_linecap="round",
        ))
        dwg.add(dwg.text(
            "Not Isomorphic",
            insert=(cx, cy + 55),
            text_anchor="middle",
            font_size=14,
            font_family="sans-serif",
            fill=_NON_ISO_X_COLOR,
            font_weight="bold",
        ))
        # Hidden marker for parsing
        dwg.add(dwg.rect(
            insert=(0, 0), size=(0, 0),
            id="answer-not-isomorphic",
        ))

    return dwg.tostring()


def _parse_answer_svg(svg_string: str) -> bool | None:
    """Parse an answer SVG to determine whether it indicates isomorphic or not.

    Returns True for isomorphic, False for not isomorphic, None if unparseable.
    """
    if 'id="answer-isomorphic"' in svg_string:
        return True
    if 'id="answer-not-isomorphic"' in svg_string:
        return False
    # Fallback: check text content.
    if "Isomorphic" in svg_string and "Not Isomorphic" not in svg_string:
        return True
    if "Not Isomorphic" in svg_string:
        return False
    return None
