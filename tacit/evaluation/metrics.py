# tacit/evaluation/metrics.py
"""Scoring and aggregation metrics for TACIT Benchmark."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_accuracy(results: list[bool]) -> float:
    """Compute overall accuracy from a list of pass/fail booleans."""
    if not results:
        return 0.0
    return sum(results) / len(results)


def compute_accuracy_by_difficulty(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute accuracy grouped by difficulty level."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        groups[r["difficulty"]].append(r["correct"])
    return {k: compute_accuracy(v) for k, v in groups.items()}


def compute_accuracy_by_task(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute accuracy grouped by task name."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        groups[r["task"]].append(r["correct"])
    return {k: compute_accuracy(v) for k, v in groups.items()}
