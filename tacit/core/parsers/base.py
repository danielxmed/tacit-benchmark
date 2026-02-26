"""Base structural parser interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """Abstract base for task-specific visual parsers.

    Track 1 (generative) evaluation extracts structure from
    model-generated PNG images.  Each task provides a parser that
    converts PNG bytes into a structural representation the verifier
    can check.
    """

    @abstractmethod
    def parse(self, png_bytes: bytes) -> Any:
        """Parse PNG image bytes into structural representation."""
        ...
