"""SVG structural parser for maze puzzles."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from tacit.core.parsers.base import BaseParser


class MazeParser(BaseParser):
    """Parse maze SVG to extract path coordinates.

    Extracts the solution/candidate path from the hidden text element
    embedded in maze SVGs by the MazeGenerator.
    """

    def parse(self, svg_string: str) -> list[tuple[int, int, int]]:
        """Extract path coordinates from SVG.

        Returns:
            List of (layer, row, col) tuples representing the path,
            or empty list if no path data found.
        """
        # Try XML parsing first
        try:
            root = ET.fromstring(svg_string)

            # Search for the hidden text element with id="maze-path"
            for elem in root.iter():
                attrib_id = elem.get("id", "")
                if attrib_id == "maze-path":
                    text = elem.text
                    if text:
                        return self._parse_path_string(text)

            # Also search with SVG namespace
            for elem in root.iter("{http://www.w3.org/2000/svg}text"):
                attrib_id = elem.get("id", "")
                if attrib_id == "maze-path":
                    text = elem.text
                    if text:
                        return self._parse_path_string(text)
        except ET.ParseError:
            pass

        # Fallback: regex extraction
        match = re.search(r'id="maze-path"[^>]*>([^<]+)<', svg_string)
        if match:
            return self._parse_path_string(match.group(1))

        return []

    @staticmethod
    def _parse_path_string(path_str: str) -> list[tuple[int, int, int]]:
        """Parse path string format 'layer,row,col;layer,row,col;...'."""
        path = []
        for cell_str in path_str.strip().split(";"):
            parts = cell_str.strip().split(",")
            if len(parts) == 3:
                path.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return path
