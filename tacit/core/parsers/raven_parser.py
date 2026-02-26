"""Structural parser for Raven's Progressive Matrices SVG output."""
from __future__ import annotations

import re
from typing import Any

from tacit.core.parsers.base import BaseParser


_ATTR_PREFIX = "data-tacit-"


class RavenParser(BaseParser):
    """Parse Raven's matrix tile SVGs into structural representation.

    Extracts tile attributes (shape, color, size, rotation, count) from
    data-tacit-* attributes embedded as HTML comments in the SVG.
    """

    def parse(self, svg_string: str) -> dict[str, Any]:
        """Parse SVG string into tile attribute dictionary.

        Returns:
            Dictionary with keys: shape, color, size, rotation, count.
            Returns empty dict if no attributes found.
        """
        attrs: dict[str, Any] = {}
        for key in ["shape", "color", "size", "rotation", "count"]:
            pattern = rf'{_ATTR_PREFIX}{key}="([^"]*)"'
            match = re.search(pattern, svg_string)
            if match:
                val: Any = match.group(1)
                if key == "rotation":
                    val = int(val)
                elif key == "count":
                    val = int(val)
                attrs[key] = val

        return attrs
