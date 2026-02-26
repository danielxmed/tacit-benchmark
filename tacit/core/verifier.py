"""Base verification interface for TACIT Benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tacit.core.types import PuzzleInstance, VerificationResult


class BaseVerifier(ABC):
    """Abstract base for task-specific verifiers."""

    @abstractmethod
    def verify(
        self, puzzle: PuzzleInstance, candidate_png: bytes
    ) -> VerificationResult:
        """Verify a candidate solution PNG against the puzzle."""
        ...

    @abstractmethod
    def extract_structure(self, png_bytes: bytes) -> Any:
        """Extract structural representation from PNG for verification."""
        ...
