"""Multi-layer maze generator for TACIT Benchmark.

Generates 2D mazes with multiple layers connected by portals.
Uses randomized DFS (recursive backtracker) for maze carving,
BFS for cross-layer pathfinding, and SVG rendering with the
shared renderer module.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any

import numpy as np

from tacit.core.renderer import (
    STYLE,
    create_canvas,
    draw_circle,
    draw_line,
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

# Cell values in the grid
WALL = 0
PASSAGE = 1

# SVG layout constants
CELL_PX = 12
LAYER_GAP_PX = 30
MARGIN_PX = 20
PORTAL_RADIUS = 4


class MazeGenerator(BaseGenerator):
    """Generator for multi-layer maze puzzles with portals."""

    def __init__(self) -> None:
        super().__init__(task_name="maze")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[dict[str, Any], list[tuple[int, int, int]]]:
        grid_size = difficulty.params.get("grid_size", 8)
        num_layers = difficulty.params.get("layers", 1)
        num_portals = difficulty.params.get("portals", 0)

        # Internal grid dimensions: 2*grid_size + 1 (walls on even indices)
        gs = 2 * grid_size + 1

        # Generate each layer
        grids: list[np.ndarray] = []
        for _ in range(num_layers):
            grid = self._carve_maze(gs, rng)
            grids.append(grid)

        # Place portals between adjacent layers
        portals = self._place_portals(grids, num_portals, num_layers, rng)

        # Start and end positions (passage cells)
        start = (0, 1, 1)  # layer 0, row 1, col 1
        end_layer = num_layers - 1
        end = (end_layer, gs - 2, gs - 2)

        # Ensure start and end are passages
        grids[start[0]][start[1], start[2]] = PASSAGE
        grids[end[0]][end[1], end[2]] = PASSAGE

        # Solve via BFS across layers
        solution_path = self._solve_bfs(grids, portals, start, end)
        if solution_path is None:
            # Fallback: if BFS fails, force a path by opening walls
            solution_path = self._force_path(grids, portals, start, end, rng)

        puzzle_data = {
            "grids": grids,
            "portals": portals,
            "start": start,
            "end": end,
            "grid_size": grid_size,
            "gs": gs,
            "num_layers": num_layers,
        }
        return puzzle_data, solution_path

    def _carve_maze(self, gs: int, rng: np.random.Generator) -> np.ndarray:
        """Carve a maze using iterative randomized DFS (recursive backtracker).

        Grid layout: even indices are walls, odd indices are passages.
        We carve between odd-indexed cells by removing the wall between them.
        """
        grid = np.zeros((gs, gs), dtype=np.int8)

        # Mark all odd-indexed cells as potential passage cells
        maze_rows = list(range(1, gs, 2))
        maze_cols = list(range(1, gs, 2))

        visited = set()
        start_r, start_c = maze_rows[0], maze_cols[0]
        stack = [(start_r, start_c)]
        visited.add((start_r, start_c))
        grid[start_r, start_c] = PASSAGE

        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

        while stack:
            r, c = stack[-1]
            # Find unvisited neighbors
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < gs - 1 and 1 <= nc < gs - 1 and (nr, nc) not in visited:
                    neighbors.append((nr, nc, r + dr // 2, c + dc // 2))

            if neighbors:
                idx = int(rng.integers(0, len(neighbors)))
                nr, nc, wr, wc = neighbors[idx]
                grid[wr, wc] = PASSAGE  # Remove wall
                grid[nr, nc] = PASSAGE  # Mark cell
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        return grid

    def _place_portals(
        self,
        grids: list[np.ndarray],
        num_portals: int,
        num_layers: int,
        rng: np.random.Generator,
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        """Place portals between adjacent layers at passage cells."""
        if num_layers < 2 or num_portals == 0:
            return []

        portals: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        gs = grids[0].shape[0]

        for _ in range(num_portals):
            # Pick a random pair of adjacent layers
            layer_a = int(rng.integers(0, num_layers - 1))
            layer_b = layer_a + 1

            # Find passage cells that are passages in both layers
            passage_cells = []
            for r in range(1, gs - 1):
                for c in range(1, gs - 1):
                    if grids[layer_a][r, c] == PASSAGE and grids[layer_b][r, c] == PASSAGE:
                        passage_cells.append((r, c))

            if passage_cells:
                idx = int(rng.integers(0, len(passage_cells)))
                r, c = passage_cells[idx]
                portal = ((layer_a, r, c), (layer_b, r, c))
                portals.append(portal)

        return portals

    def _solve_bfs(
        self,
        grids: list[np.ndarray],
        portals: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
        start: tuple[int, int, int],
        end: tuple[int, int, int],
    ) -> list[tuple[int, int, int]] | None:
        """BFS pathfinding across layers using portals as edges."""
        # Build portal lookup
        portal_map: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for a, b in portals:
            portal_map[a] = b
            portal_map[b] = a

        queue: deque[tuple[tuple[int, int, int], list[tuple[int, int, int]]]] = deque()
        queue.append((start, [start]))
        visited: set[tuple[int, int, int]] = {start}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue:
            (layer, r, c), path = queue.popleft()

            if (layer, r, c) == end:
                return path

            grid = grids[layer]
            gs = grid.shape[0]

            # Move within same layer
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs and grid[nr, nc] == PASSAGE:
                    state = (layer, nr, nc)
                    if state not in visited:
                        visited.add(state)
                        queue.append((state, path + [state]))

            # Use portal if available
            pos = (layer, r, c)
            if pos in portal_map:
                dest = portal_map[pos]
                if dest not in visited:
                    visited.add(dest)
                    queue.append((dest, path + [dest]))

        return None

    def _force_path(
        self,
        grids: list[np.ndarray],
        portals: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        rng: np.random.Generator,
    ) -> list[tuple[int, int, int]]:
        """Force a path from start to end by opening walls if needed.

        This ensures every generated maze is solvable. For multi-layer
        mazes, we open passages on each layer independently, then use
        portals to transition between layers.
        """
        num_layers = len(grids)
        gs = grids[0].shape[0]

        if num_layers == 1:
            # Force direct path on single layer
            path = self._force_single_layer_path(
                grids[0], (start[1], start[2]), (end[1], end[2])
            )
            return [(0, r, c) for r, c in path]

        # Multi-layer: ensure portal connectivity
        # Force path from start to first portal on layer 0
        # Then from portal exits to next portal on subsequent layers
        # Finally from last portal to end on last layer

        # Build portal lookup by layer
        portal_by_layer: dict[int, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {}
        for a, b in portals:
            la = a[0]
            portal_by_layer.setdefault(la, []).append((a, b))
            lb = b[0]
            portal_by_layer.setdefault(lb, []).append((b, a))

        full_path: list[tuple[int, int, int]] = []
        current = start

        for layer in range(num_layers - 1):
            # Find a portal on this layer going to layer+1
            layer_portals = portal_by_layer.get(layer, [])
            target_portal = None
            for a, b in layer_portals:
                if a[0] == layer and b[0] == layer + 1:
                    target_portal = (a, b)
                    break

            if target_portal is None:
                # Force create a portal at a passage cell
                r, c = gs // 2 | 1, gs // 2 | 1  # Ensure odd
                grids[layer][r, c] = PASSAGE
                grids[layer + 1][r, c] = PASSAGE
                target_portal = ((layer, r, c), (layer + 1, r, c))
                portals.append(target_portal)

            portal_entry = target_portal[0]
            portal_exit = target_portal[1]

            # Force path from current to portal entry on this layer
            seg = self._force_single_layer_path(
                grids[layer],
                (current[1], current[2]),
                (portal_entry[1], portal_entry[2]),
            )
            for r, c in seg:
                pos = (layer, r, c)
                if not full_path or pos != full_path[-1]:
                    full_path.append(pos)

            # Cross portal
            full_path.append(portal_exit)
            current = portal_exit

        # Force path from current to end on last layer
        seg = self._force_single_layer_path(
            grids[num_layers - 1],
            (current[1], current[2]),
            (end[1], end[2]),
        )
        for r, c in seg:
            pos = (num_layers - 1, r, c)
            if not full_path or pos != full_path[-1]:
                full_path.append(pos)

        return full_path

    def _force_single_layer_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Force a path on a single layer by opening walls as needed."""
        gs = grid.shape[0]
        r, c = start
        tr, tc = end
        path = [(r, c)]
        grid[r, c] = PASSAGE

        # Simple L-shaped path: go rows first, then columns
        while r != tr:
            step = 1 if tr > r else -1
            r += step
            grid[r, c] = PASSAGE
            path.append((r, c))

        while c != tc:
            step = 1 if tc > c else -1
            c += step
            grid[r, c] = PASSAGE
            path.append((r, c))

        return path

    # ------------------------------------------------------------------
    # SVG rendering
    # ------------------------------------------------------------------

    def _generate_puzzle_svg(self, puzzle_data: dict[str, Any]) -> str:
        """Render maze grids without solution path."""
        return self._render_svg(puzzle_data, solution_path=None)

    def _generate_solution_svg(
        self, puzzle_data: dict[str, Any], solution_data: list[tuple[int, int, int]]
    ) -> str:
        """Render maze grids with solution path overlay."""
        return self._render_svg(puzzle_data, solution_path=solution_data)

    def _render_svg(
        self,
        puzzle_data: dict[str, Any],
        solution_path: list[tuple[int, int, int]] | None = None,
    ) -> str:
        """Render all layers side-by-side with optional path overlay."""
        grids = puzzle_data["grids"]
        portals = puzzle_data["portals"]
        start = puzzle_data["start"]
        end = puzzle_data["end"]
        gs = puzzle_data["gs"]
        num_layers = puzzle_data["num_layers"]

        layer_width = gs * CELL_PX
        total_width = (
            MARGIN_PX * 2
            + num_layers * layer_width
            + (num_layers - 1) * LAYER_GAP_PX
        )
        total_height = MARGIN_PX * 2 + gs * CELL_PX + 20  # 20 for label

        canvas = create_canvas(total_width, total_height)

        for layer_idx in range(num_layers):
            x_offset = MARGIN_PX + layer_idx * (layer_width + LAYER_GAP_PX)
            y_offset = MARGIN_PX

            self._render_layer(canvas, grids[layer_idx], x_offset, y_offset, gs)

            # Layer label
            draw_text(
                canvas,
                x_offset + layer_width / 2,
                y_offset + gs * CELL_PX + 16,
                f"Layer {layer_idx + 1}",
                font_size=11,
            )

        # Render portals
        for idx, (a, b) in enumerate(portals):
            color = STYLE["colors"][idx % len(STYLE["colors"])]
            for layer, r, c in [a, b]:
                x_off = MARGIN_PX + layer * (layer_width + LAYER_GAP_PX)
                y_off = MARGIN_PX
                cx = x_off + c * CELL_PX + CELL_PX / 2
                cy = y_off + r * CELL_PX + CELL_PX / 2
                draw_circle(canvas, cx, cy, PORTAL_RADIUS, fill=color, stroke=color)

        # Render start and end markers
        for marker, color, label in [
            (start, "#00CC00", "S"),
            (end, "#CC0000", "E"),
        ]:
            layer, r, c = marker
            x_off = MARGIN_PX + layer * (layer_width + LAYER_GAP_PX)
            y_off = MARGIN_PX
            cx = x_off + c * CELL_PX + CELL_PX / 2
            cy = y_off + r * CELL_PX + CELL_PX / 2
            draw_circle(canvas, cx, cy, CELL_PX / 2 - 1, fill=color, stroke=color)
            draw_text(canvas, cx, cy + 4, label, font_size=9, fill="#FFFFFF")

        # Render solution path if provided
        if solution_path is not None:
            self._render_path(canvas, solution_path, puzzle_data, STYLE["solution_color"])
            # Embed path data for verification as a hidden text element
            path_data = ";".join(
                f"{layer},{r},{c}" for layer, r, c in solution_path
            )
            text_elem = canvas.text(
                path_data,
                insert=(0, 0),
                visibility="hidden",
                id="maze-path",
                font_size="1px",
            )
            canvas.add(text_elem)

        return svg_to_string(canvas)

    def _render_layer(
        self,
        canvas: Any,
        grid: np.ndarray,
        x_offset: float,
        y_offset: float,
        gs: int,
    ) -> None:
        """Render a single layer's walls."""
        # Draw border
        draw_rect(
            canvas,
            x_offset,
            y_offset,
            gs * CELL_PX,
            gs * CELL_PX,
            fill="none",
        )

        # Draw walls as filled rectangles
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] == WALL:
                    draw_rect(
                        canvas,
                        x_offset + c * CELL_PX,
                        y_offset + r * CELL_PX,
                        CELL_PX,
                        CELL_PX,
                        fill=STYLE["line_color"],
                        stroke=STYLE["line_color"],
                        stroke_width=0.5,
                    )

    def _render_path(
        self,
        canvas: Any,
        path: list[tuple[int, int, int]],
        puzzle_data: dict[str, Any],
        color: str,
    ) -> None:
        """Render a path as connected line segments."""
        gs = puzzle_data["gs"]
        layer_width = gs * CELL_PX

        for i in range(len(path) - 1):
            l1, r1, c1 = path[i]
            l2, r2, c2 = path[i + 1]

            x_off1 = MARGIN_PX + l1 * (layer_width + LAYER_GAP_PX)
            y_off1 = MARGIN_PX
            x_off2 = MARGIN_PX + l2 * (layer_width + LAYER_GAP_PX)
            y_off2 = MARGIN_PX

            x1 = x_off1 + c1 * CELL_PX + CELL_PX / 2
            y1 = y_off1 + r1 * CELL_PX + CELL_PX / 2
            x2 = x_off2 + c2 * CELL_PX + CELL_PX / 2
            y2 = y_off2 + r2 * CELL_PX + CELL_PX / 2

            draw_line(canvas, x1, y1, x2, y2, stroke=color, stroke_width=2.5)

    # ------------------------------------------------------------------
    # Distractor generation
    # ------------------------------------------------------------------

    def _available_violations(self) -> list[str]:
        return ["wall_breach", "portal_skip", "disconnected", "wrong_exit"]

    def _generate_distractor(
        self,
        puzzle_data: dict[str, Any],
        solution_data: list[tuple[int, int, int]],
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a distractor path with exactly one violation type."""
        path = list(solution_data)  # copy

        if violation_type == "wall_breach":
            path = self._distract_wall_breach(puzzle_data, path, rng)
        elif violation_type == "portal_skip":
            path = self._distract_portal_skip(puzzle_data, path, rng)
        elif violation_type == "disconnected":
            path = self._distract_disconnected(puzzle_data, path, rng)
        elif violation_type == "wrong_exit":
            path = self._distract_wrong_exit(puzzle_data, path, rng)

        svg = self._render_svg(puzzle_data, solution_path=path)
        return svg, violation_type

    def _distract_wall_breach(
        self,
        puzzle_data: dict[str, Any],
        path: list[tuple[int, int, int]],
        rng: np.random.Generator,
    ) -> list[tuple[int, int, int]]:
        """Create a path that goes through a wall cell."""
        grids = puzzle_data["grids"]
        gs = puzzle_data["gs"]

        if len(path) < 3:
            return path

        # Pick a random segment of the path and reroute through a wall
        attempts = 0
        while attempts < 50:
            idx = int(rng.integers(1, len(path) - 1))
            layer, r, c = path[idx]
            grid = grids[layer]

            # Find an adjacent wall cell
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs and grid[nr, nc] == WALL:
                    # Insert wall cell into path
                    new_path = path[:idx] + [(layer, nr, nc)] + path[idx:]
                    return new_path
            attempts += 1

        # Fallback: insert a known wall cell
        wall_cell = self._find_wall_cell(grids[0], gs)
        if wall_cell:
            idx = max(1, len(path) // 2)
            return path[:idx] + [(0, wall_cell[0], wall_cell[1])] + path[idx:]
        return path

    def _distract_portal_skip(
        self,
        puzzle_data: dict[str, Any],
        path: list[tuple[int, int, int]],
        rng: np.random.Generator,
    ) -> list[tuple[int, int, int]]:
        """Create a path that skips a required portal (layer transition without portal)."""
        num_layers = puzzle_data["num_layers"]
        gs = puzzle_data["gs"]

        if num_layers < 2:
            # No portals to skip; fall back to wall_breach
            return self._distract_wall_breach(puzzle_data, path, rng)

        # Find a layer transition in the path and remove the portal step
        for i in range(len(path) - 1):
            if path[i][0] != path[i + 1][0]:
                # This is a portal crossing; replace with a direct jump
                # (same row/col but different layer without portal)
                layer_from = path[i][0]
                layer_to = path[i + 1][0]
                r, c = path[i][1], path[i][2]
                # Move to a different cell on the target layer
                new_r = min(r + 2, gs - 2) if r + 2 < gs - 1 else max(r - 2, 1)
                new_path = path[:i + 1] + [(layer_to, new_r, c)] + path[i + 2:]
                return new_path

        # No layer transition found; fall back to wall_breach
        return self._distract_wall_breach(puzzle_data, path, rng)

    def _distract_disconnected(
        self,
        puzzle_data: dict[str, Any],
        path: list[tuple[int, int, int]],
        rng: np.random.Generator,
    ) -> list[tuple[int, int, int]]:
        """Create a path with a gap (non-adjacent consecutive cells)."""
        if len(path) < 4:
            return path

        # Remove a segment from the middle
        gap_start = int(rng.integers(1, len(path) // 2))
        gap_end = min(gap_start + int(rng.integers(2, 5)), len(path) - 1)

        return path[:gap_start] + path[gap_end:]

    def _distract_wrong_exit(
        self,
        puzzle_data: dict[str, Any],
        path: list[tuple[int, int, int]],
        rng: np.random.Generator,
    ) -> list[tuple[int, int, int]]:
        """Create a path that ends at the wrong cell."""
        grids = puzzle_data["grids"]
        end = puzzle_data["end"]
        gs = puzzle_data["gs"]
        end_layer = end[0]

        # Find a different passage cell on the end layer as wrong exit
        grid = grids[end_layer]
        candidates = []
        for r in range(1, gs - 1):
            for c in range(1, gs - 1):
                if grid[r, c] == PASSAGE and (end_layer, r, c) != end:
                    candidates.append((end_layer, r, c))

        if candidates:
            idx = int(rng.integers(0, len(candidates)))
            wrong_end = candidates[idx]
            # Replace last cell with wrong exit
            return path[:-1] + [wrong_end]

        return path

    @staticmethod
    def _find_wall_cell(
        grid: np.ndarray, gs: int
    ) -> tuple[int, int] | None:
        """Find any wall cell in the grid interior."""
        for r in range(1, gs - 1):
            for c in range(1, gs - 1):
                if grid[r, c] == WALL:
                    return (r, c)
        return None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify a candidate solution SVG against the maze.

        Extracts the path from the hidden text element in the SVG,
        then checks:
        1. Path starts at the maze start
        2. Path ends at the maze end
        3. All cells in the path are passages (no wall breaches)
        4. Consecutive cells are adjacent or connected by a portal
        """
        # Extract path data from SVG
        candidate_path = self._extract_path_from_svg(candidate_svg)
        if candidate_path is None:
            return VerificationResult(passed=False, reason="No path data found in SVG")

        # Reconstruct puzzle data from the original puzzle SVG
        original_path = self._extract_path_from_svg(puzzle.solution_svg)
        if original_path is None:
            return VerificationResult(
                passed=False, reason="Cannot extract reference path"
            )

        # Re-generate puzzle data for full verification
        # We need grids, portals, start, end from the puzzle
        # Since we embed path data, we can verify structurally
        puzzle_data = self._reconstruct_puzzle_data(puzzle)

        if puzzle_data is None:
            # Fallback: compare paths directly
            if candidate_path == original_path:
                return VerificationResult(passed=True)
            return VerificationResult(passed=False, reason="Path mismatch")

        grids = puzzle_data["grids"]
        portals = puzzle_data["portals"]
        start = puzzle_data["start"]
        end = puzzle_data["end"]

        # Check start
        if not candidate_path or candidate_path[0] != start:
            return VerificationResult(
                passed=False,
                reason=f"Path does not start at {start}, starts at {candidate_path[0] if candidate_path else 'empty'}",
            )

        # Check end
        if candidate_path[-1] != end:
            return VerificationResult(
                passed=False,
                reason=f"Path does not end at {end}, ends at {candidate_path[-1]}",
            )

        # Build portal lookup
        portal_set: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
        for a, b in portals:
            portal_set.add((a, b))
            portal_set.add((b, a))

        # Check each step
        for i in range(len(candidate_path)):
            layer, r, c = candidate_path[i]

            # Check cell is a passage
            if layer < 0 or layer >= len(grids):
                return VerificationResult(
                    passed=False,
                    reason=f"Invalid layer {layer} at step {i}",
                )
            gs = grids[layer].shape[0]
            if r < 0 or r >= gs or c < 0 or c >= gs:
                return VerificationResult(
                    passed=False,
                    reason=f"Out of bounds ({layer},{r},{c}) at step {i}",
                )
            if grids[layer][r, c] != PASSAGE:
                return VerificationResult(
                    passed=False,
                    reason=f"Wall breach at ({layer},{r},{c}) step {i}",
                )

            # Check adjacency with previous cell
            if i > 0:
                prev = candidate_path[i - 1]
                if not self._is_adjacent_or_portal(prev, candidate_path[i], portal_set):
                    return VerificationResult(
                        passed=False,
                        reason=f"Disconnected path at step {i}: {prev} -> {candidate_path[i]}",
                    )

        return VerificationResult(passed=True)

    @staticmethod
    def _is_adjacent_or_portal(
        a: tuple[int, int, int],
        b: tuple[int, int, int],
        portal_set: set[tuple[tuple[int, int, int], tuple[int, int, int]]],
    ) -> bool:
        """Check if two cells are adjacent on the same layer or connected by a portal."""
        if a[0] == b[0]:
            # Same layer: check Manhattan distance == 1
            return abs(a[1] - b[1]) + abs(a[2] - b[2]) == 1
        # Different layers: must be a portal
        return (a, b) in portal_set

    def _extract_path_from_svg(
        self, svg_string: str
    ) -> list[tuple[int, int, int]] | None:
        """Extract path data from the hidden text element in SVG."""
        # Try XML parsing first
        try:
            # Register SVG namespace to avoid prefix issues
            namespaces = {"svg": "http://www.w3.org/2000/svg"}
            root = ET.fromstring(svg_string)

            # Search for the hidden text element with id="maze-path"
            for elem in root.iter():
                attrib_id = elem.get("id", "")
                if attrib_id == "maze-path":
                    text = elem.text
                    if text:
                        return self._parse_path_string(text)

            # Also search with namespace
            for elem in root.iter("{http://www.w3.org/2000/svg}text"):
                attrib_id = elem.get("id", "")
                if attrib_id == "maze-path":
                    text = elem.text
                    if text:
                        return self._parse_path_string(text)
        except ET.ParseError:
            pass

        # Fallback: regex extraction
        match = re.search(r'id="maze-path"[^>]*>([^<]+)<', svg_string)
        if match:
            return self._parse_path_string(match.group(1))

        return None

    @staticmethod
    def _parse_path_string(
        path_str: str,
    ) -> list[tuple[int, int, int]]:
        """Parse path string format 'layer,row,col;layer,row,col;...'."""
        path = []
        for cell_str in path_str.strip().split(";"):
            parts = cell_str.strip().split(",")
            if len(parts) == 3:
                path.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return path

    def _reconstruct_puzzle_data(
        self, puzzle: PuzzleInstance
    ) -> dict[str, Any] | None:
        """Reconstruct puzzle data by re-generating with same seed.

        This ensures we have access to grids and portals for verification.
        """
        try:
            rng = np.random.default_rng(puzzle.seed)
            puzzle_data, _ = self._generate_puzzle(puzzle.difficulty, rng)
            return puzzle_data
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Difficulty axes
    # ------------------------------------------------------------------

    def difficulty_axes(self) -> list[DifficultyRange]:
        return [
            DifficultyRange("grid_size", 4, 128, step=4),
            DifficultyRange("layers", 1, 8, step=1),
            DifficultyRange("portals", 0, 20, step=1),
        ]
