"""Base structural parser interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """Abstract base for task-specific SVG structural parsers.

    Track 1 (generative) evaluation needs to extract structure from
    model-generated images. Each task provides a parser that converts
    SVG/image data into a structural representation the verifier can check.
    """

    @abstractmethod
    def parse(self, svg_string: str) -> Any:
        """Parse SVG string into structural representation."""
        ...
