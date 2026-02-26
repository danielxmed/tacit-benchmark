# tacit/evaluation/harness.py
"""Core evaluation orchestration for TACIT Benchmark."""
from __future__ import annotations

from typing import Any

from tacit.generators.base import BaseGenerator


# Registry of task name -> generator class
_GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {}


def register_generator(task_name: str, generator_cls: type[BaseGenerator]) -> None:
    """Register a generator class for a task."""
    _GENERATOR_REGISTRY[task_name] = generator_cls


class EvaluationHarness:
    """Orchestrates evaluation across tasks and tracks."""

    def __init__(self) -> None:
        self._generators: dict[str, BaseGenerator] = {}
        self._load_generators()

    def _load_generators(self) -> None:
        """Instantiate all registered generators."""
        for name, cls in _GENERATOR_REGISTRY.items():
            self._generators[name] = cls()

    def available_tasks(self) -> list[str]:
        """List available task names."""
        return list(self._generators.keys())

    def get_generator(self, task_name: str) -> BaseGenerator:
        """Get generator instance by task name."""
        return self._generators[task_name]

    def run_track2(
        self,
        task_name: str,
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Run Track 2 evaluation for a task. Returns accuracy metrics."""
        from tacit.evaluation.track2 import evaluate_discriminative
        from tacit.evaluation.metrics import compute_accuracy

        corrects = []
        for r in results:
            res = evaluate_discriminative(r["correct_index"], r["selected_index"])
            corrects.append(res.correct)
        return {"accuracy": compute_accuracy(corrects), "total": len(corrects)}
