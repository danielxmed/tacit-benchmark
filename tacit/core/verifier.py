"""Base verification interface for TACIT Benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tacit.core.types import PuzzleInstance, VerificationResult


class BaseVerifier(ABC):
    """Abstract base for task-specific verifiers.

    Each task implements:
    - extract_structure: parse SVG into a structural representation
    - verify: check if a candidate solution is correct
    """

    @abstractmethod
    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify a candidate solution against the puzzle."""
        ...

    @abstractmethod
    def extract_structure(self, svg_string: str) -> Any:
        """Extract structural representation from SVG for verification."""
        ...
