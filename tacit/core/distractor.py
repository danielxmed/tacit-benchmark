"""Distractor generation framework for TACIT Benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDistractorGenerator(ABC):
    """Abstract base for task-specific distractor generators.

    Distractors are near-miss solutions that violate exactly one
    structural constraint. Each task defines its own violation types.
    """

    @abstractmethod
    def generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a single distractor SVG.

        Returns:
            (distractor_svg, violation_type)
        """
        ...

    @abstractmethod
    def available_violations(self) -> list[str]:
        """List the violation types this task supports."""
        ...

    def generate_set(
        self,
        puzzle_data: Any,
        solution_data: Any,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[list[str], list[str]]:
        """Generate a set of distractors with diverse violation types.

        Cycles through available violation types to ensure diversity.
        """
        violations = self.available_violations()
        svgs: list[str] = []
        violation_labels: list[str] = []
        for i in range(count):
            vtype = violations[i % len(violations)]
            svg, label = self.generate_distractor(
                puzzle_data, solution_data, vtype, rng,
            )
            svgs.append(svg)
            violation_labels.append(label)
        return svgs, violation_labels
