"""Cellular Automata Forward Prediction generator.

Given an initial grid at state T and a set of transition rules,
predict the grid state at T+k by simulating k steps of an
outer totalistic 2D cellular automaton (Moore neighborhood).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tacit.core.types import (
    DifficultyParams,
    DifficultyRange,
    PuzzleInstance,
    VerificationResult,
)
from tacit.generators.base import BaseGenerator
from tacit.generators._ca_common import (
    generate_initial_grid,
    generate_rule,
    grid_to_svg,
    render_forward_puzzle_svg,
    simulate,
)


class CAForwardGenerator(BaseGenerator):
    """Generator for CA forward prediction puzzles.

    Puzzle: initial grid + transition rule table (visual).
    Solution: the grid after k simulation steps.
    Verification: deterministic simulation comparison.
    """

    def __init__(self) -> None:
        super().__init__(task_name="ca_forward")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        grid_size: int = difficulty.params.get("grid_size", 8)
        rule_complexity: int = difficulty.params.get("rule_complexity", 2)
        steps: int = difficulty.params.get("steps", 1)

        num_states = max(2, min(rule_complexity, 8))

        rule = generate_rule(num_states, rng)
        initial_grid = generate_initial_grid(grid_size, num_states, rng)
        final_grid = simulate(initial_grid, rule, steps)

        puzzle_data = {
            "initial_grid": initial_grid,
            "rule": rule,
            "num_states": num_states,
            "grid_size": grid_size,
            "steps": steps,
        }
        solution_data = {
            "final_grid": final_grid,
        }
        return puzzle_data, solution_data

    # ------------------------------------------------------------------
    # SVG rendering
    # ------------------------------------------------------------------

    def _generate_puzzle_svg(self, puzzle_data: dict[str, Any]) -> str:
        return render_forward_puzzle_svg(
            puzzle_data["initial_grid"],
            puzzle_data["rule"],
            puzzle_data["steps"],
        )

    def _generate_solution_svg(
        self, puzzle_data: dict[str, Any], solution_data: dict[str, Any]
    ) -> str:
        return grid_to_svg(
            solution_data["final_grid"],
            title=f"State T+{puzzle_data['steps']}",
        )

    # ------------------------------------------------------------------
    # Distractors
    # ------------------------------------------------------------------

    def _available_violations(self) -> list[str]:
        return ["wrong_cell", "wrong_step_count", "wrong_rule"]

    def _generate_distractor(
        self,
        puzzle_data: dict[str, Any],
        solution_data: dict[str, Any],
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        grid_size = puzzle_data["grid_size"]
        num_states = puzzle_data["num_states"]
        steps = puzzle_data["steps"]
        initial_grid = puzzle_data["initial_grid"]
        rule = puzzle_data["rule"]
        final_grid = solution_data["final_grid"]

        if violation_type == "wrong_cell":
            distractor = self._distract_wrong_cells(
                final_grid, num_states, rng
            )
        elif violation_type == "wrong_step_count":
            distractor = self._distract_wrong_steps(
                initial_grid, rule, steps, rng
            )
        elif violation_type == "wrong_rule":
            distractor = self._distract_wrong_rule(
                initial_grid, rule, num_states, steps, rng
            )
        else:
            distractor = self._distract_wrong_cells(
                final_grid, num_states, rng
            )

        svg = grid_to_svg(distractor, title=f"State T+{steps}")
        return svg, violation_type

    def _distract_wrong_cells(
        self,
        correct_grid: np.ndarray,
        num_states: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Flip a few random cells to wrong states."""
        distractor = correct_grid.copy()
        rows, cols = distractor.shape
        num_flips = max(1, rng.integers(1, max(2, rows * cols // 8)))
        for _ in range(num_flips):
            r = rng.integers(0, rows)
            c = rng.integers(0, cols)
            old_val = int(distractor[r, c])
            candidates = [s for s in range(num_states) if s != old_val]
            if candidates:
                distractor[r, c] = rng.choice(candidates)
        return distractor

    def _distract_wrong_steps(
        self,
        initial_grid: np.ndarray,
        rule: np.ndarray,
        correct_steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate the wrong number of steps."""
        if correct_steps > 1:
            wrong_steps = rng.choice(
                [s for s in range(1, correct_steps + 3) if s != correct_steps]
            )
        else:
            wrong_steps = rng.choice([2, 3])
        return simulate(initial_grid, rule, int(wrong_steps))

    def _distract_wrong_rule(
        self,
        initial_grid: np.ndarray,
        correct_rule: np.ndarray,
        num_states: int,
        steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply a different rule to the initial grid."""
        wrong_rule = generate_rule(num_states, rng)
        return simulate(initial_grid, wrong_rule, steps)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self, puzzle: PuzzleInstance, candidate_png: bytes
    ) -> VerificationResult:
        """Verify candidate PNG shows the correct grid at state T+k."""
        # Regenerate expected solution
        rng = np.random.default_rng(puzzle.seed)
        puzzle_data, solution_data = self._generate_puzzle(puzzle.difficulty, rng)
        expected_grid = solution_data["final_grid"]
        grid_size = expected_grid.shape[0]

        # Parse candidate PNG
        from tacit.generators._ca_common import parse_grid_from_png

        candidate_grid = parse_grid_from_png(candidate_png, grid_size)
        if candidate_grid is None:
            return VerificationResult(
                passed=False,
                reason="Could not parse grid from candidate PNG.",
            )

        if np.array_equal(candidate_grid, expected_grid):
            return VerificationResult(passed=True)

        diff_count = int(np.sum(candidate_grid != expected_grid))
        return VerificationResult(
            passed=False,
            reason=f"Grid mismatch: {diff_count} of {grid_size * grid_size} cells differ.",
            details={"diff_count": diff_count},
        )

    # ------------------------------------------------------------------
    # Difficulty axes
    # ------------------------------------------------------------------

    def difficulty_axes(self) -> list[DifficultyRange]:
        return [
            DifficultyRange(
                name="grid_size",
                min_val=4,
                max_val=64,
                step=4,
                description="Width/height of the square grid",
            ),
            DifficultyRange(
                name="rule_complexity",
                min_val=2,
                max_val=8,
                step=1,
                description="Number of cell states (2=binary, 8=max)",
            ),
            DifficultyRange(
                name="steps",
                min_val=1,
                max_val=20,
                step=1,
                description="Number of simulation steps to predict",
            ),
        ]
