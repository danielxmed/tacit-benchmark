# tests/evaluation/test_harness.py
import pytest
from unittest.mock import MagicMock, patch


def test_harness_loads_generators():
    from tacit.evaluation.harness import EvaluationHarness
    harness = EvaluationHarness()
    generators = harness.available_tasks()
    assert isinstance(generators, list)


def test_harness_run_track2():
    from tacit.evaluation.harness import EvaluationHarness
    harness = EvaluationHarness()
    # Minimal smoke test -- full integration tested via CLI
    assert hasattr(harness, "run_track2")


def test_harness_run_track2_returns_metrics():
    """run_track2 should return accuracy metrics dict."""
    from tacit.evaluation.harness import EvaluationHarness, register_generator
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import VerificationResult, DifficultyRange

    # Create a minimal mock generator class
    class _MockGen(BaseGenerator):
        def __init__(self) -> None:
            super().__init__(task_name="_test_task")

        def _generate_puzzle(self, difficulty, rng):
            return {}, {}

        def _generate_puzzle_svg(self, puzzle_data):
            return "<svg></svg>"

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return "<svg></svg>"

        def _generate_distractor(self, puzzle_data, solution_data, vtype, rng):
            return "<svg></svg>", vtype

        def _available_violations(self):
            return ["a"]

        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="x", min_val=1, max_val=10)]

    register_generator("_test_task", _MockGen)
    try:
        harness = EvaluationHarness()
        results_data = [
            {"correct_index": 0, "selected_index": 0},
            {"correct_index": 1, "selected_index": 2},
        ]
        metrics = harness.run_track2("_test_task", results_data)
        assert "accuracy" in metrics
        assert metrics["accuracy"] == pytest.approx(0.5)
        assert metrics["total"] == 2
    finally:
        # Clean up registry
        from tacit.evaluation import harness as h_mod
        h_mod._GENERATOR_REGISTRY.pop("_test_task", None)


def test_harness_get_generator():
    """get_generator should return the instantiated generator."""
    from tacit.evaluation.harness import EvaluationHarness, register_generator
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import VerificationResult, DifficultyRange

    class _MockGen2(BaseGenerator):
        def __init__(self) -> None:
            super().__init__(task_name="_test_task2")

        def _generate_puzzle(self, difficulty, rng):
            return {}, {}

        def _generate_puzzle_svg(self, puzzle_data):
            return "<svg></svg>"

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return "<svg></svg>"

        def _generate_distractor(self, puzzle_data, solution_data, vtype, rng):
            return "<svg></svg>", vtype

        def _available_violations(self):
            return ["a"]

        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="x", min_val=1, max_val=10)]

    register_generator("_test_task2", _MockGen2)
    try:
        harness = EvaluationHarness()
        gen = harness.get_generator("_test_task2")
        assert isinstance(gen, _MockGen2)
    finally:
        from tacit.evaluation import harness as h_mod
        h_mod._GENERATOR_REGISTRY.pop("_test_task2", None)


def test_harness_get_generator_key_error():
    """get_generator should raise KeyError for unknown task."""
    from tacit.evaluation.harness import EvaluationHarness
    harness = EvaluationHarness()
    with pytest.raises(KeyError):
        harness.get_generator("nonexistent_task_xyz")


def test_register_generator():
    """register_generator should add entry to the registry."""
    from tacit.evaluation.harness import register_generator, _GENERATOR_REGISTRY
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import VerificationResult, DifficultyRange

    class _MockGen3(BaseGenerator):
        def __init__(self) -> None:
            super().__init__(task_name="_test_task3")

        def _generate_puzzle(self, difficulty, rng):
            return {}, {}

        def _generate_puzzle_svg(self, puzzle_data):
            return "<svg></svg>"

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return "<svg></svg>"

        def _generate_distractor(self, puzzle_data, solution_data, vtype, rng):
            return "<svg></svg>", vtype

        def _available_violations(self):
            return ["a"]

        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="x", min_val=1, max_val=10)]

    register_generator("_test_task3", _MockGen3)
    try:
        assert "_test_task3" in _GENERATOR_REGISTRY
        assert _GENERATOR_REGISTRY["_test_task3"] is _MockGen3
    finally:
        _GENERATOR_REGISTRY.pop("_test_task3", None)
