"""Cellular Automata Inverse Inference generator.

Given the grid at state T and state T+k, infer the transition rule
that was applied. The solution is the rule table rendered visually.
Verification: apply the inferred rule to state T for k steps and
confirm it produces state T+k.
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
    parse_rule_from_png,
    parse_rule_from_svg,
    render_inverse_puzzle_svg,
    render_rule_table_svg,
    simulate,
)


class CAInverseGenerator(BaseGenerator):
    """Generator for CA inverse inference puzzles.

    Puzzle: state T grid + state T+k grid shown side by side.
    Solution: the transition rule table (visual).
    Verification: parse rule from candidate SVG, simulate k steps from T,
                  compare to T+k cell-by-cell.
    """

    def __init__(self) -> None:
        super().__init__(task_name="ca_inverse")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        grid_size: int = difficulty.params.get("grid_size", 8)
        rule_space: int = difficulty.params.get("rule_space", 4)
        steps: int = difficulty.params.get("steps", 1)

        num_states = max(2, min(rule_space, 8))

        rule = generate_rule(num_states, rng)
        initial_grid = generate_initial_grid(grid_size, num_states, rng)
        final_grid = simulate(initial_grid, rule, steps)

        puzzle_data = {
            "initial_grid": initial_grid,
            "final_grid": final_grid,
            "num_states": num_states,
            "grid_size": grid_size,
            "steps": steps,
        }
        solution_data = {
            "rule": rule,
        }
        return puzzle_data, solution_data

    # ------------------------------------------------------------------
    # SVG rendering
    # ------------------------------------------------------------------

    def _generate_puzzle_svg(self, puzzle_data: dict[str, Any]) -> str:
        return render_inverse_puzzle_svg(
            puzzle_data["initial_grid"],
            puzzle_data["final_grid"],
            puzzle_data["steps"],
        )

    def _generate_solution_svg(
        self, puzzle_data: dict[str, Any], solution_data: dict[str, Any]
    ) -> str:
        return render_rule_table_svg(
            solution_data["rule"],
            title="Transition Rule",
        )

    # ------------------------------------------------------------------
    # Distractors
    # ------------------------------------------------------------------

    def _available_violations(self) -> list[str]:
        return ["off_by_one_rule", "transposed_rule", "partial_rule"]

    def _generate_distractor(
        self,
        puzzle_data: dict[str, Any],
        solution_data: dict[str, Any],
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        rule = solution_data["rule"]
        num_states = puzzle_data["num_states"]

        if violation_type == "off_by_one_rule":
            distractor_rule = self._distract_off_by_one(rule, num_states, rng)
        elif violation_type == "transposed_rule":
            distractor_rule = self._distract_transposed(rule, rng)
        elif violation_type == "partial_rule":
            distractor_rule = self._distract_partial(rule, num_states, rng)
        else:
            distractor_rule = self._distract_off_by_one(rule, num_states, rng)

        svg = render_rule_table_svg(distractor_rule, title="Transition Rule")
        return svg, violation_type

    def _distract_off_by_one(
        self,
        correct_rule: np.ndarray,
        num_states: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Change one entry in the rule table to a wrong state."""
        distractor = correct_rule.copy()
        rows, cols = distractor.shape
        r = rng.integers(0, rows)
        c = rng.integers(0, cols)
        old_val = int(distractor[r, c])
        candidates = [s for s in range(num_states) if s != old_val]
        if candidates:
            distractor[r, c] = rng.choice(candidates)
        return distractor

    def _distract_transposed(
        self,
        correct_rule: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Swap two entries in the rule table that have different values."""
        distractor = correct_rule.copy()
        rows, cols = distractor.shape
        # Find two positions with different values to ensure a real change
        attempts = 0
        while attempts < 50:
            r1 = int(rng.integers(0, rows))
            c1 = int(rng.integers(0, cols))
            r2 = int(rng.integers(0, rows))
            c2 = int(rng.integers(0, cols))
            if (r1 != r2 or c1 != c2) and distractor[r1, c1] != distractor[r2, c2]:
                break
            attempts += 1
        distractor[r1, c1], distractor[r2, c2] = (
            distractor[r2, c2],
            distractor[r1, c1],
        )
        # Fallback: if the swap still produced identical rule (unlikely),
        # flip one cell to guarantee difference
        if np.array_equal(distractor, correct_rule):
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            old_val = int(distractor[r, c])
            num_states = rows  # rows == num_states in rule table
            candidates = [s for s in range(num_states) if s != old_val]
            if candidates:
                distractor[r, c] = rng.choice(candidates)
        return distractor

    def _distract_partial(
        self,
        correct_rule: np.ndarray,
        num_states: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Change multiple entries — rule works for some cells but not all."""
        distractor = correct_rule.copy()
        rows, cols = distractor.shape
        num_changes = max(2, rng.integers(2, max(3, rows * cols // 4)))
        for _ in range(num_changes):
            r = rng.integers(0, rows)
            c = rng.integers(0, cols)
            old_val = int(distractor[r, c])
            candidates = [s for s in range(num_states) if s != old_val]
            if candidates:
                distractor[r, c] = rng.choice(candidates)
        return distractor

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self, puzzle: PuzzleInstance, candidate_png: bytes
    ) -> VerificationResult:
        """Verify candidate PNG shows the correct rule table."""
        # Regenerate expected rule
        rng = np.random.default_rng(puzzle.seed)
        _puzzle_data, solution_data = self._generate_puzzle(puzzle.difficulty, rng)
        expected_rule = solution_data["rule"]

        rule_space = puzzle.difficulty.params.get("rule_space", 4)
        num_states = max(2, min(rule_space, 8))

        # Parse candidate PNG — cell_size=16 matches render_rule_table_svg default
        candidate_rule = parse_rule_from_png(candidate_png, num_states, cell_size=16)
        if candidate_rule is None:
            return VerificationResult(
                passed=False,
                reason="Could not parse rule table from candidate PNG.",
            )

        if np.array_equal(candidate_rule, expected_rule):
            return VerificationResult(passed=True)

        diff_count = int(np.sum(candidate_rule != expected_rule))
        total = expected_rule.size
        return VerificationResult(
            passed=False,
            reason=f"Rule mismatch: {diff_count} of {total} entries differ.",
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
                name="rule_space",
                min_val=2,
                max_val=8,
                step=1,
                description="Number of cell states in the rule space",
            ),
            DifficultyRange(
                name="steps",
                min_val=1,
                max_val=20,
                step=1,
                description="Number of simulation steps between T and T+k",
            ),
        ]
