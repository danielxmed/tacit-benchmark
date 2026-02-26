# tacit/evaluation/report.py
"""Result report generation for TACIT Benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_report(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Write evaluation results as JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
