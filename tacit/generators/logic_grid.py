"""Visual Logic Grid Generator for TACIT Benchmark.

Generates constraint-based logic grid puzzles using only symbols and colors
as clues (no natural language). The puzzle is based on Latin square generation
with constraint-based solving.

Constraint types (all visual, no text):
- row_unique / col_unique: colored marker in cell border
- symbol_adjacency: connector symbol between cells
- color_exclusion: X mark meaning "this symbol cannot be here"
- equality: = mark between cells meaning "same category"

Generation approach:
1. Start with a valid Latin square solution
2. Generate constraints that uniquely determine it
3. Verify uniqueness via backtracking solver
4. Distractors violate exactly one constraint
"""
from __future__ import annotations

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

# Visual symbols for grid cell content (no text, pure visual markers)
SYMBOLS = [
    "\u25CF",  # Black circle
    "\u25A0",  # Black square
    "\u25B2",  # Black triangle up
    "\u2666",  # Black diamond
    "\u2605",  # Black star
    "\u2B22",  # Black hexagon
    "\u25C6",  # Black diamond (alt)
    "\u2B23",  # Horizontal hexagon
    "\u25CB",  # White circle
    "\u25A1",  # White square
]

# Constraint visual markers
CONSTRAINT_MARKERS = {
    "row_unique": "\u2192",     # Right arrow (row indicator)
    "col_unique": "\u2193",     # Down arrow (column indicator)
    "exclusion": "\u2715",      # Multiplication X (cannot be here)
    "adjacency": "\u2194",      # Left-right arrow (adjacency)
    "equality": "=",            # Equality sign
}


class LogicGridGenerator(BaseGenerator):
    """Generator for visual logic grid puzzles.

    Creates Latin square-based puzzles with visual-only constraint clues.
    """

    def __init__(self) -> None:
        super().__init__(task_name="logic_grid")

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate a logic grid puzzle and its solution.

        Returns:
            (puzzle_data, solution_data) where puzzle_data contains the grid
            size, constraints, and visual clue info; solution_data contains
            the completed Latin square grid.
        """
        grid_size = difficulty.params.get("grid_size", 4)
        num_constraints = difficulty.params.get("constraints", 6)
        num_types = difficulty.params.get("types", 2)

        # Generate a valid Latin square as the solution
        solution_grid = self._generate_latin_square(grid_size, rng)

        # Generate constraints that uniquely identify this solution
        constraints = self._generate_constraints(
            solution_grid, num_constraints, num_types, rng
        )

        # Assign colors to symbols for visual rendering
        color_indices = list(range(grid_size))
        rng.shuffle(color_indices)
        symbol_colors = {i: STYLE["colors"][color_indices[i] % len(STYLE["colors"])] for i in range(grid_size)}

        puzzle_data = {
            "grid_size": grid_size,
            "constraints": constraints,
            "symbol_colors": symbol_colors,
            "solution_grid": solution_grid,
        }

        solution_data = {
            "grid": solution_grid,
        }

        return puzzle_data, solution_data

    def _generate_latin_square(
        self, n: int, rng: np.random.Generator
    ) -> list[list[int]]:
        """Generate a random valid Latin square of size n.

        Uses a permutation-based approach: start with the canonical
        Latin square and apply random row/column/symbol permutations.
        """
        # Start with canonical Latin square: row i, col j -> (i + j) % n
        grid = [[(i + j) % n for j in range(n)] for i in range(n)]

        # Permute rows
        row_perm = list(range(n))
        rng.shuffle(row_perm)
        grid = [grid[row_perm[i]] for i in range(n)]

        # Permute columns
        col_perm = list(range(n))
        rng.shuffle(col_perm)
        grid = [[row[col_perm[j]] for j in range(n)] for row in grid]

        # Permute symbols
        sym_perm = list(range(n))
        rng.shuffle(sym_perm)
        grid = [[sym_perm[cell] for cell in row] for row in grid]

        return grid

    def _generate_constraints(
        self,
        grid: list[list[int]],
        num_constraints: int,
        num_types: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Generate constraints that uniquely identify the solution.

        Generates a mix of constraint types, then verifies the constraint
        set yields a unique solution via backtracking. If not unique,
        adds more constraints until uniqueness is achieved.
        """
        n = len(grid)
        all_constraint_types = ["row_unique", "col_unique", "exclusion", "adjacency"]
        # Use only the requested number of types
        active_types = all_constraint_types[:max(1, min(num_types, len(all_constraint_types)))]

        constraints: list[dict[str, Any]] = []
        candidate_constraints = self._enumerate_possible_constraints(grid, active_types, rng)

        # Shuffle candidates for randomness
        indices = list(range(len(candidate_constraints)))
        rng.shuffle(indices)
        candidate_constraints = [candidate_constraints[i] for i in indices]

        # Add constraints until we have enough and solution is unique
        for c in candidate_constraints:
            constraints.append(c)
            if len(constraints) >= num_constraints:
                if self._is_unique_solution(grid, constraints):
                    break

        # If still not unique, keep adding
        if not self._is_unique_solution(grid, constraints):
            for c in candidate_constraints:
                if c not in constraints:
                    constraints.append(c)
                    if self._is_unique_solution(grid, constraints):
                        break

        return constraints

    def _enumerate_possible_constraints(
        self,
        grid: list[list[int]],
        active_types: list[str],
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Enumerate possible constraints from the solution grid."""
        n = len(grid)
        candidates: list[dict[str, Any]] = []

        for ctype in active_types:
            if ctype == "row_unique":
                # "Symbol X is in row R" constraints
                for r in range(n):
                    for c in range(n):
                        candidates.append({
                            "type": "row_unique",
                            "row": r,
                            "symbol": grid[r][c],
                            "col": c,
                        })

            elif ctype == "col_unique":
                # "Symbol X is in column C" constraints
                for r in range(n):
                    for c in range(n):
                        candidates.append({
                            "type": "col_unique",
                            "col": c,
                            "symbol": grid[r][c],
                            "row": r,
                        })

            elif ctype == "exclusion":
                # "Symbol X is NOT at position (r, c)" constraints
                for r in range(n):
                    for c in range(n):
                        for s in range(n):
                            if grid[r][c] != s:
                                candidates.append({
                                    "type": "exclusion",
                                    "row": r,
                                    "col": c,
                                    "symbol": s,
                                })

            elif ctype == "adjacency":
                # "Cells (r1,c1) and (r2,c2) have/don't have same symbol"
                for r in range(n):
                    for c in range(n - 1):
                        candidates.append({
                            "type": "adjacency",
                            "row1": r, "col1": c,
                            "row2": r, "col2": c + 1,
                            "same": grid[r][c] == grid[r][c + 1],
                        })
                for r in range(n - 1):
                    for c in range(n):
                        candidates.append({
                            "type": "adjacency",
                            "row1": r, "col1": c,
                            "row2": r + 1, "col2": c,
                            "same": grid[r][c] == grid[r + 1][c],
                        })

        return candidates

    def _is_unique_solution(
        self,
        target_grid: list[list[int]],
        constraints: list[dict[str, Any]],
    ) -> bool:
        """Check if constraints yield exactly one valid solution via backtracking.

        Returns True if the target grid is the only solution satisfying
        all constraints.
        """
        n = len(target_grid)
        # Use backtracking to count solutions (stop at 2)
        count = self._count_solutions(n, constraints, max_count=2)
        return count == 1

    def _count_solutions(
        self,
        n: int,
        constraints: list[dict[str, Any]],
        max_count: int = 2,
    ) -> int:
        """Count solutions using backtracking, stopping at max_count."""
        grid = [[-1] * n for _ in range(n)]
        result = [0]

        def is_valid_placement(row: int, col: int, val: int) -> bool:
            """Check if placing val at (row, col) is consistent."""
            # Latin square: no duplicate in row or column
            for c in range(n):
                if grid[row][c] == val:
                    return False
            for r in range(n):
                if grid[r][col] == val:
                    return False

            # Check constraints
            for c in constraints:
                if not self._check_partial_constraint(grid, n, row, col, val, c):
                    return False

            return True

        def solve(pos: int) -> None:
            if result[0] >= max_count:
                return
            if pos == n * n:
                result[0] += 1
                return

            row, col = pos // n, pos % n
            for val in range(n):
                if is_valid_placement(row, col, val):
                    grid[row][col] = val
                    solve(pos + 1)
                    grid[row][col] = -1
                    if result[0] >= max_count:
                        return

        solve(0)
        return result[0]

    def _check_partial_constraint(
        self,
        grid: list[list[int]],
        n: int,
        row: int,
        col: int,
        val: int,
        constraint: dict[str, Any],
    ) -> bool:
        """Check if placing val at (row, col) is consistent with a constraint.

        For partially filled grids, only check when the constraint's
        cells are known.
        """
        ctype = constraint["type"]

        if ctype == "row_unique":
            # Symbol must be at specific position in row
            c_row = constraint["row"]
            c_sym = constraint["symbol"]
            c_col = constraint["col"]
            if row == c_row and col == c_col and val != c_sym:
                return False
            if row == c_row and col != c_col and val == c_sym:
                return False

        elif ctype == "col_unique":
            c_col = constraint["col"]
            c_sym = constraint["symbol"]
            c_row = constraint["row"]
            if col == c_col and row == c_row and val != c_sym:
                return False
            if col == c_col and row != c_row and val == c_sym:
                return False

        elif ctype == "exclusion":
            c_row = constraint["row"]
            c_col = constraint["col"]
            c_sym = constraint["symbol"]
            if row == c_row and col == c_col and val == c_sym:
                return False

        elif ctype == "adjacency":
            r1, c1 = constraint["row1"], constraint["col1"]
            r2, c2 = constraint["row2"], constraint["col2"]
            same = constraint["same"]

            # Only check when both cells are filled
            if row == r1 and col == c1:
                other = grid[r2][c2]
                if other != -1:
                    if same and val != other:
                        return False
                    if not same and val == other:
                        return False
            elif row == r2 and col == c2:
                other = grid[r1][c1]
                if other != -1:
                    if same and val != other:
                        return False
                    if not same and val == other:
                        return False

        return True

    def _check_full_constraint(
        self,
        grid: list[list[int]],
        constraint: dict[str, Any],
    ) -> bool:
        """Check if a fully filled grid satisfies a constraint."""
        ctype = constraint["type"]

        if ctype == "row_unique":
            return grid[constraint["row"]][constraint["col"]] == constraint["symbol"]

        elif ctype == "col_unique":
            return grid[constraint["row"]][constraint["col"]] == constraint["symbol"]

        elif ctype == "exclusion":
            return grid[constraint["row"]][constraint["col"]] != constraint["symbol"]

        elif ctype == "adjacency":
            r1, c1 = constraint["row1"], constraint["col1"]
            r2, c2 = constraint["row2"], constraint["col2"]
            same = constraint["same"]
            if same:
                return grid[r1][c1] == grid[r2][c2]
            else:
                return grid[r1][c1] != grid[r2][c2]

        return True

    def _generate_puzzle_svg(self, puzzle_data: dict[str, Any]) -> str:
        """Render the puzzle grid with constraint clues as SVG."""
        n = puzzle_data["grid_size"]
        constraints = puzzle_data["constraints"]
        symbol_colors = puzzle_data["symbol_colors"]

        cell_size = 60
        margin = 80
        clue_margin = 30
        canvas_size = margin + n * cell_size + clue_margin + margin

        canvas = create_canvas(canvas_size, canvas_size)

        # Draw grid
        for r in range(n):
            for c in range(n):
                x = margin + c * cell_size
                y = margin + r * cell_size
                draw_rect(canvas, x, y, cell_size, cell_size,
                          fill="#FAFAFA", stroke=STYLE["grid_color"])

        # Draw outer border
        draw_rect(canvas, margin, margin, n * cell_size, n * cell_size,
                  fill="none", stroke=STYLE["line_color"], stroke_width=3)

        # Render constraint clues visually
        self._render_constraint_clues(canvas, constraints, symbol_colors,
                                      n, cell_size, margin)

        # Draw symbol legend at top
        self._render_symbol_legend(canvas, n, symbol_colors, margin, cell_size)

        return svg_to_string(canvas)

    def _render_constraint_clues(
        self,
        canvas: Any,
        constraints: list[dict[str, Any]],
        symbol_colors: dict[int, str],
        n: int,
        cell_size: int,
        margin: int,
    ) -> None:
        """Render visual constraint clues on the puzzle grid."""
        for constraint in constraints:
            ctype = constraint["type"]

            if ctype == "row_unique":
                row = constraint["row"]
                col = constraint["col"]
                symbol = constraint["symbol"]
                color = symbol_colors[symbol]
                # Draw colored marker at the cell edge
                cx = margin + col * cell_size + cell_size / 2
                cy = margin + row * cell_size + 8
                draw_circle(canvas, cx, cy, 4, fill=color, stroke=color)
                # Draw arrow marker in left margin
                mx = margin - 20
                my = margin + row * cell_size + cell_size / 2
                draw_text(canvas, mx, my + 4, CONSTRAINT_MARKERS["row_unique"],
                          font_size=12, fill=color)

            elif ctype == "col_unique":
                row = constraint["row"]
                col = constraint["col"]
                symbol = constraint["symbol"]
                color = symbol_colors[symbol]
                # Draw colored marker at cell edge
                cx = margin + col * cell_size + cell_size - 8
                cy = margin + row * cell_size + cell_size / 2
                draw_circle(canvas, cx, cy, 4, fill=color, stroke=color)
                # Draw arrow marker in top margin
                mx = margin + col * cell_size + cell_size / 2
                my = margin - 10
                draw_text(canvas, mx, my, CONSTRAINT_MARKERS["col_unique"],
                          font_size=12, fill=color)

            elif ctype == "exclusion":
                row = constraint["row"]
                col = constraint["col"]
                symbol = constraint["symbol"]
                color = symbol_colors[symbol]
                # Draw X mark in the cell with symbol color
                cx = margin + col * cell_size + cell_size / 2
                cy = margin + row * cell_size + cell_size / 2
                draw_text(canvas, cx, cy + 4, CONSTRAINT_MARKERS["exclusion"],
                          font_size=14, fill=color)

            elif ctype == "adjacency":
                r1, c1 = constraint["row1"], constraint["col1"]
                r2, c2 = constraint["row2"], constraint["col2"]
                same = constraint["same"]
                # Draw connector between cells
                x1 = margin + c1 * cell_size + cell_size / 2
                y1 = margin + r1 * cell_size + cell_size / 2
                x2 = margin + c2 * cell_size + cell_size / 2
                y2 = margin + r2 * cell_size + cell_size / 2
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                marker = "=" if same else "\u2260"
                color = STYLE["highlight_color"] if same else STYLE["line_color"]
                draw_text(canvas, mid_x, mid_y + 4, marker,
                          font_size=14, fill=color)

    def _render_symbol_legend(
        self,
        canvas: Any,
        n: int,
        symbol_colors: dict[int, str],
        margin: int,
        cell_size: int,
    ) -> None:
        """Render a symbol legend showing which color maps to which symbol."""
        legend_y = margin - 40
        total_width = n * 40
        start_x = margin + (n * cell_size - total_width) / 2

        for i in range(n):
            x = start_x + i * 40 + 20
            color = symbol_colors[i]
            symbol = SYMBOLS[i % len(SYMBOLS)]
            draw_text(canvas, x, legend_y, symbol, font_size=16, fill=color)

    def _generate_solution_svg(
        self, puzzle_data: dict[str, Any], solution_data: dict[str, Any]
    ) -> str:
        """Render the complete solution grid as SVG."""
        n = puzzle_data["grid_size"]
        grid = solution_data["grid"]
        symbol_colors = puzzle_data["symbol_colors"]

        cell_size = 60
        margin = 80
        clue_margin = 30
        canvas_size = margin + n * cell_size + clue_margin + margin

        canvas = create_canvas(canvas_size, canvas_size)

        # Draw grid with symbols filled in
        for r in range(n):
            for c in range(n):
                x = margin + c * cell_size
                y = margin + r * cell_size
                draw_rect(canvas, x, y, cell_size, cell_size,
                          fill="#FAFAFA", stroke=STYLE["grid_color"])

                # Draw the symbol
                val = grid[r][c]
                symbol = SYMBOLS[val % len(SYMBOLS)]
                color = symbol_colors[val]
                cx = x + cell_size / 2
                cy = y + cell_size / 2 + 5
                draw_text(canvas, cx, cy, symbol, font_size=20, fill=color)

        # Draw outer border
        draw_rect(canvas, margin, margin, n * cell_size, n * cell_size,
                  fill="none", stroke=STYLE["line_color"], stroke_width=3)

        return svg_to_string(canvas)

    def _grid_to_svg(
        self,
        grid: list[list[int]],
        n: int,
        symbol_colors: dict[int, str],
    ) -> str:
        """Convert a grid to SVG (helper for distractor generation)."""
        cell_size = 60
        margin = 80
        clue_margin = 30
        canvas_size = margin + n * cell_size + clue_margin + margin

        canvas = create_canvas(canvas_size, canvas_size)

        for r in range(n):
            for c in range(n):
                x = margin + c * cell_size
                y = margin + r * cell_size
                draw_rect(canvas, x, y, cell_size, cell_size,
                          fill="#FAFAFA", stroke=STYLE["grid_color"])

                val = grid[r][c]
                symbol = SYMBOLS[val % len(SYMBOLS)]
                color = symbol_colors[val]
                cx = x + cell_size / 2
                cy = y + cell_size / 2 + 5
                draw_text(canvas, cx, cy, symbol, font_size=20, fill=color)

        draw_rect(canvas, margin, margin, n * cell_size, n * cell_size,
                  fill="none", stroke=STYLE["line_color"], stroke_width=3)

        return svg_to_string(canvas)

    def _generate_distractor(
        self,
        puzzle_data: dict[str, Any],
        solution_data: dict[str, Any],
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a single distractor with a specific violation type."""
        n = puzzle_data["grid_size"]
        grid = solution_data["grid"]
        symbol_colors = puzzle_data["symbol_colors"]
        constraints = puzzle_data["constraints"]

        if violation_type == "constraint_violation":
            distractor_grid = self._make_constraint_violation(grid, constraints, n, rng)
        elif violation_type == "symbol_swap":
            distractor_grid = self._make_symbol_swap(grid, n, rng)
        elif violation_type == "non_unique":
            distractor_grid = self._make_non_unique(grid, constraints, n, rng)
        else:
            # Default to symbol swap
            distractor_grid = self._make_symbol_swap(grid, n, rng)

        svg = self._grid_to_svg(distractor_grid, n, symbol_colors)
        return svg, violation_type

    def _make_constraint_violation(
        self,
        grid: list[list[int]],
        constraints: list[dict[str, Any]],
        n: int,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Create a grid that violates exactly one constraint.

        Picks a random constraint and modifies the grid to violate it
        while trying to keep it a valid Latin square otherwise.
        """
        new_grid = [row[:] for row in grid]

        if not constraints:
            # Fallback: just swap two cells in a row
            return self._make_symbol_swap(grid, n, rng)

        # Pick a random constraint to violate
        idx = int(rng.integers(0, len(constraints)))
        constraint = constraints[idx]
        ctype = constraint["type"]

        if ctype in ("row_unique", "col_unique"):
            # Swap the specified cell with another in the same row
            row = constraint["row"]
            col = constraint["col"]
            other_col = int(rng.integers(0, n))
            while other_col == col:
                other_col = int(rng.integers(0, n))
            new_grid[row][col], new_grid[row][other_col] = (
                new_grid[row][other_col], new_grid[row][col]
            )

        elif ctype == "exclusion":
            # Place the excluded symbol at the specified position
            row = constraint["row"]
            col = constraint["col"]
            excluded_sym = constraint["symbol"]
            # Find where the excluded symbol is in this row
            for c in range(n):
                if new_grid[row][c] == excluded_sym:
                    new_grid[row][c], new_grid[row][col] = (
                        new_grid[row][col], new_grid[row][c]
                    )
                    break

        elif ctype == "adjacency":
            r1, c1 = constraint["row1"], constraint["col1"]
            r2, c2 = constraint["row2"], constraint["col2"]
            same = constraint["same"]
            if same:
                # They should be same but we make them different
                # Swap one cell with another in its row
                other_col = int(rng.integers(0, n))
                while other_col == c1 or new_grid[r1][other_col] == new_grid[r2][c2]:
                    other_col = int(rng.integers(0, n))
                new_grid[r1][c1], new_grid[r1][other_col] = (
                    new_grid[r1][other_col], new_grid[r1][c1]
                )
            else:
                # They should be different but we make them same
                # This is tricky in a Latin square; swap to create equality
                val2 = new_grid[r2][c2]
                for c in range(n):
                    if new_grid[r1][c] == val2:
                        new_grid[r1][c], new_grid[r1][c1] = (
                            new_grid[r1][c1], new_grid[r1][c]
                        )
                        break

        return new_grid

    def _make_symbol_swap(
        self,
        grid: list[list[int]],
        n: int,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Swap two symbols in the same row (maintains row validity but breaks columns)."""
        new_grid = [row[:] for row in grid]
        row = int(rng.integers(0, n))
        c1 = int(rng.integers(0, n))
        c2 = int(rng.integers(0, n))
        while c2 == c1:
            c2 = int(rng.integers(0, n))
        new_grid[row][c1], new_grid[row][c2] = new_grid[row][c2], new_grid[row][c1]
        return new_grid

    def _make_non_unique(
        self,
        grid: list[list[int]],
        constraints: list[dict[str, Any]],
        n: int,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Create a grid that is a valid Latin square but not the intended solution.

        Generates a different Latin square. Since constraints should uniquely
        determine the solution, this alternative will violate at least one constraint.
        """
        # Generate a different Latin square by shuffling
        new_grid = self._generate_latin_square(n, rng)

        # Make sure it's actually different from the solution
        attempts = 0
        while new_grid == grid and attempts < 20:
            new_grid = self._generate_latin_square(n, rng)
            attempts += 1

        return new_grid

    def _extract_grid_from_svg(self, svg_string: str, n: int) -> list[list[int]] | None:
        """Extract the grid values from an SVG string by parsing symbol text.

        Returns the grid as a list of lists of symbol indices, or None if
        parsing fails.
        """
        grid = [[-1] * n for _ in range(n)]

        # Parse SVG to find text elements with symbols
        # We look for our known symbols in the SVG text
        import re

        cell_size = 60
        margin = 80

        # Find all text elements
        text_pattern = re.compile(
            r'<text[^>]*?x="([^"]+)"[^>]*?y="([^"]+)"[^>]*?>([^<]+)</text>'
        )
        # Also match insert="(x, y)" format from svgwrite
        insert_pattern = re.compile(
            r'<text[^>]*?insert="([^"]+)"[^>]*?>([^<]+)</text>'
        )

        matches = text_pattern.findall(svg_string)
        insert_matches = insert_pattern.findall(svg_string)

        # Combine matches
        all_text_items: list[tuple[float, float, str]] = []

        for x_str, y_str, text in matches:
            try:
                x = float(x_str)
                y = float(y_str)
                all_text_items.append((x, y, text))
            except ValueError:
                continue

        for insert_str, text in insert_matches:
            try:
                # Parse "x, y" or "(x, y)"
                coords = insert_str.strip("()").split(",")
                x = float(coords[0].strip())
                y = float(coords[1].strip())
                all_text_items.append((x, y, text))
            except (ValueError, IndexError):
                continue

        # Map text items to grid positions
        for x, y, text in all_text_items:
            text = text.strip()
            # Check if this is one of our symbols
            symbol_idx = None
            for idx, sym in enumerate(SYMBOLS):
                if text == sym:
                    symbol_idx = idx
                    break

            if symbol_idx is None:
                continue

            # Calculate grid position from coordinates
            col = round((x - margin - cell_size / 2) / cell_size)
            row = round((y - margin - cell_size / 2 - 5) / cell_size)

            if 0 <= row < n and 0 <= col < n:
                grid[row][col] = symbol_idx

        # Validate: all cells must be filled
        for r in range(n):
            for c in range(n):
                if grid[r][c] == -1:
                    return None

        return grid

    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify a candidate solution against the puzzle constraints.

        Extracts the grid from the candidate SVG and checks:
        1. Valid Latin square (unique symbols per row and column)
        2. All constraints are satisfied
        """
        n = puzzle.difficulty.params.get("grid_size", 4)

        # Extract grid from candidate SVG
        candidate_grid = self._extract_grid_from_svg(candidate_svg, n)

        if candidate_grid is None:
            return VerificationResult(
                passed=False,
                reason="Could not parse grid from SVG",
            )

        # Extract expected solution grid from the puzzle's solution SVG
        expected_grid = self._extract_grid_from_svg(puzzle.solution_svg, n)

        if expected_grid is None:
            return VerificationResult(
                passed=False,
                reason="Could not parse expected solution from SVG",
            )

        # Check if grids match (symbol indices should match)
        if candidate_grid == expected_grid:
            return VerificationResult(
                passed=True,
                reason="Solution matches expected grid",
            )

        # If grids don't match exactly, check if candidate satisfies all constraints
        # First check Latin square property
        for r in range(n):
            if len(set(candidate_grid[r])) != n:
                return VerificationResult(
                    passed=False,
                    reason=f"Row {r} has duplicate symbols",
                    details={"row": r, "values": candidate_grid[r]},
                )

        for c in range(n):
            col_vals = [candidate_grid[r][c] for r in range(n)]
            if len(set(col_vals)) != n:
                return VerificationResult(
                    passed=False,
                    reason=f"Column {c} has duplicate symbols",
                    details={"col": c, "values": col_vals},
                )

        # Check all constraints from puzzle metadata
        constraints = puzzle.metadata.get("constraints", [])
        for i, constraint in enumerate(constraints):
            if not self._check_full_constraint(candidate_grid, constraint):
                return VerificationResult(
                    passed=False,
                    reason=f"Constraint {i} ({constraint['type']}) violated",
                    details={"constraint_index": i, "constraint": constraint},
                )

        return VerificationResult(
            passed=True,
            reason="All constraints satisfied",
        )

    def _available_violations(self) -> list[str]:
        """List the violation types this task supports."""
        return ["constraint_violation", "symbol_swap", "non_unique"]

    def difficulty_axes(self) -> list[DifficultyRange]:
        """Declare difficulty parameters for logic grid puzzles."""
        return [
            DifficultyRange(
                name="grid_size",
                min_val=3,
                max_val=8,
                step=1,
                description="Size of the NxN logic grid",
            ),
            DifficultyRange(
                name="constraints",
                min_val=4,
                max_val=20,
                step=1,
                description="Number of constraint clues",
            ),
            DifficultyRange(
                name="types",
                min_val=1,
                max_val=4,
                step=1,
                description="Number of distinct constraint types used",
            ),
        ]

    def generate(
        self,
        difficulty: DifficultyParams,
        seed: int,
        num_distractors: int = 4,
    ) -> PuzzleInstance:
        """Generate a complete logic grid puzzle instance.

        Overrides base to store extra metadata (solution_grid, constraints).
        """
        instance = super().generate(difficulty, seed, num_distractors)

        # Re-generate puzzle data to populate metadata
        # (needed because base.generate doesn't expose puzzle_data)
        rng = np.random.default_rng(seed)
        puzzle_data, solution_data = self._generate_puzzle(difficulty, rng)

        instance.metadata["solution_grid"] = solution_data["grid"]
        instance.metadata["constraints"] = puzzle_data["constraints"]
        instance.metadata["symbol_colors"] = puzzle_data["symbol_colors"]

        return instance
