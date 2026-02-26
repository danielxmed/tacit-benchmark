"""Graph coloring SVG parser for TACIT Benchmark.

Extracts node-to-color mappings from graph coloring SVG representations.
Used by the graph coloring verifier to check candidate solutions.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from tacit.core.parsers.base import BaseParser

# Default node radius used in graph coloring SVG rendering
_DEFAULT_NODE_RADIUS = 16


class GraphColoringParser(BaseParser):
    """Parses graph coloring SVGs to extract node -> color mappings.

    Supports two parsing strategies:
    1. Primary: Circle elements with id="node-{N}" attributes (reliable).
    2. Fallback: Proximity-based matching of text labels to circles
       (for externally generated SVGs without id attributes).
    """

    def parse(self, svg_string: str) -> dict[int, str]:
        """Parse SVG string into node -> fill color mapping.

        Args:
            svg_string: SVG string containing colored graph nodes.

        Returns:
            Dict mapping node_id (int) -> fill color (hex string).
        """
        root = ET.fromstring(svg_string)

        node_colors: dict[int, str] = {}

        # Primary strategy: use id="node-{N}" attributes on circles
        for elem in root.iter():
            tag = elem.tag
            if "}" in tag:
                tag = tag.split("}", 1)[1]

            if tag == "circle":
                elem_id = elem.get("id", "")
                if elem_id.startswith("node-"):
                    try:
                        node_id = int(elem_id[5:])
                        fill = elem.get("fill", "#DDDDDD")
                        node_colors[node_id] = fill
                    except ValueError:
                        pass

        if node_colors:
            return node_colors

        # Fallback: proximity-based matching
        circles: list[tuple[float, float, str]] = []
        texts: list[tuple[float, float, str]] = []

        for elem in root.iter():
            tag = elem.tag
            if "}" in tag:
                tag = tag.split("}", 1)[1]

            if tag == "circle":
                cx = float(elem.get("cx", 0))
                cy = float(elem.get("cy", 0))
                fill = elem.get("fill", "#DDDDDD")
                circles.append((cx, cy, fill))

            elif tag == "text":
                x = float(elem.get("x", "0"))
                y = float(elem.get("y", "0"))
                text_content = elem.text or ""
                texts.append((x, y, text_content.strip()))

        for tx, ty, label in texts:
            if not label.isdigit():
                continue
            node_id = int(label)
            best_dist = float("inf")
            best_fill = "#DDDDDD"
            for cx, cy, fill in circles:
                dist = (tx - cx) ** 2 + (ty - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_fill = fill
            if best_dist < (_DEFAULT_NODE_RADIUS * 3) ** 2:
                node_colors[node_id] = best_fill

        return node_colors
