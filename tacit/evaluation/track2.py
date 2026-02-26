# tacit/evaluation/track2.py
"""Track 2 -- Discriminative evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscriminativeResult:
    correct: bool
    selected_index: int
    correct_index: int


def evaluate_discriminative(
    correct_index: int,
    selected_index: int,
) -> DiscriminativeResult:
    """Evaluate a discriminative (multiple-choice) response."""
    return DiscriminativeResult(
        correct=(correct_index == selected_index),
        selected_index=selected_index,
        correct_index=correct_index,
    )
