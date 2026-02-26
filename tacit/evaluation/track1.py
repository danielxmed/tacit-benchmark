# tacit/evaluation/track1.py
"""Track 1 -- Generative evaluation."""
from __future__ import annotations

from tacit.core.types import PuzzleInstance, VerificationResult
from tacit.generators.base import BaseGenerator


def evaluate_generative(
    generator: BaseGenerator,
    puzzle: PuzzleInstance,
    candidate_png: bytes,
) -> VerificationResult:
    """Evaluate a generative (image-output) response.

    The candidate is a PNG image (bytes) produced by the model.
    Delegates to the task-specific generator's verify() method.
    """
    return generator.verify(puzzle, candidate_png)
