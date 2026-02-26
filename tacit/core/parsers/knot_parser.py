"""Structural parser for knot diagram SVG output.

Extracts the binary classification label ('unknot' or 'knot') from an
answer-badge SVG produced by the UnknotGenerator.
"""
from __future__ import annotations

from typing import Any

from tacit.core.parsers.base import BaseParser


class KnotParser(BaseParser):
    """Parse knot answer SVG into a structural classification label.

    The answer SVG contains a text element with either 'unknot' or 'knot'.
    This parser extracts that label as a string.
    """

    def parse(self, svg_string: str) -> Any:
        """Extract classification label from answer SVG.

        Returns
        -------
        dict
            ``{"label": "unknot"}`` or ``{"label": "knot"}``.
            Returns ``{"label": None}`` if parsing fails.
        """
        lower = svg_string.lower()
        if ">unknot<" in lower:
            return {"label": "unknot"}
        if ">knot<" in lower:
            return {"label": "knot"}
        return {"label": None}
