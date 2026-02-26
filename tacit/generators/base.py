"""Base generator class for TACIT Benchmark tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from tacit.core.types import (
    DifficultyParams,
    DifficultyRange,
    PuzzleInstance,
    VerificationResult,
)


class BaseGenerator(ABC):
    """Abstract base class for all task generators.

    Subclasses implement the task-specific logic:
    - _generate_puzzle: create puzzle + solution data structures
    - _generate_puzzle_svg: render puzzle to SVG
    - _generate_solution_svg: render solution to SVG
    - _generate_distractor: create a single near-miss distractor
    - _available_violations: list violation types
    - verify: check if a candidate is correct
    - difficulty_axes: declare difficulty parameters
    """

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    def generate(
        self,
        difficulty: DifficultyParams,
        seed: int,
        num_distractors: int = 4,
    ) -> PuzzleInstance:
        """Generate a complete puzzle instance.

        Uses separate RNG streams for puzzle generation and distractor
        generation to ensure puzzle determinism regardless of distractor count.
        """
        rng_puzzle = np.random.default_rng(seed)
        puzzle_data, solution_data = self._generate_puzzle(difficulty, rng_puzzle)

        puzzle_svg = self._generate_puzzle_svg(puzzle_data)
        solution_svg = self._generate_solution_svg(puzzle_data, solution_data)

        # Separate RNG for distractors so puzzle is seed-stable
        rng_distractor = np.random.default_rng(seed + 2**31)
        distractor_svgs: list[str] = []
        distractor_violations: list[str] = []
        violations = self._available_violations()

        for i in range(num_distractors):
            vtype = violations[i % len(violations)]
            svg, label = self._generate_distractor(
                puzzle_data, solution_data, vtype, rng_distractor
            )
            distractor_svgs.append(svg)
            distractor_violations.append(label)

        puzzle_id = f"{self.task_name}_{difficulty.level}_{seed:04d}"

        return PuzzleInstance(
            task=self.task_name,
            puzzle_id=puzzle_id,
            seed=seed,
            difficulty=difficulty,
            puzzle_svg=puzzle_svg,
            solution_svg=solution_svg,
            distractor_svgs=distractor_svgs,
            distractor_violations=distractor_violations,
            metadata={},
        )

    @abstractmethod
    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[Any, Any]:
        """Generate puzzle data and solution data.

        Returns:
            (puzzle_data, solution_data) — internal representations
        """
        ...

    @abstractmethod
    def _generate_puzzle_svg(self, puzzle_data: Any) -> str:
        """Render puzzle data to SVG string."""
        ...

    @abstractmethod
    def _generate_solution_svg(
        self, puzzle_data: Any, solution_data: Any
    ) -> str:
        """Render solution data to SVG string."""
        ...

    @abstractmethod
    def _generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a single distractor.

        Returns:
            (distractor_svg, violation_label)
        """
        ...

    @abstractmethod
    def _available_violations(self) -> list[str]:
        """List the violation types this task supports."""
        ...

    @abstractmethod
    def verify(
        self, puzzle: PuzzleInstance, candidate_png: bytes
    ) -> VerificationResult:
        """Verify a candidate solution (PNG image bytes)."""
        ...

    @abstractmethod
    def difficulty_axes(self) -> list[DifficultyRange]:
        """Declare difficulty parameters for this task."""
        ...
