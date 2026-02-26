# tests/generators/test_base.py
import pytest
import numpy as np


def test_base_generator_is_abstract():
    from tacit.generators.base import BaseGenerator
    with pytest.raises(TypeError):
        BaseGenerator(task_name="test")


def test_base_generator_subclass_must_implement():
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class IncompleteGen(BaseGenerator):
        pass

    with pytest.raises(TypeError):
        IncompleteGen(task_name="test")


def test_base_generator_full_subclass():
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class DummyGen(BaseGenerator):
        def _generate_puzzle(self, difficulty, rng):
            return {"grid": [[0]]}, {"path": [0]}

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return "<svg>solution</svg>"

        def _generate_puzzle_svg(self, puzzle_data):
            return "<svg>puzzle</svg>"

        def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return f"<svg>distractor:{violation_type}</svg>", violation_type

        def _available_violations(self):
            return ["type_a"]

        def verify(self, puzzle, candidate_png):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="size", min_val=4, max_val=64)]

    gen = DummyGen(task_name="dummy")
    assert gen.task_name == "dummy"


def test_base_generator_generate():
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class DummyGen(BaseGenerator):
        def _generate_puzzle(self, difficulty, rng):
            return {"grid": [[0]]}, {"path": [0]}

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return "<svg>solution</svg>"

        def _generate_puzzle_svg(self, puzzle_data):
            return "<svg>puzzle</svg>"

        def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return f"<svg>d</svg>", violation_type

        def _available_violations(self):
            return ["type_a"]

        def verify(self, puzzle, candidate_png):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="size", min_val=4, max_val=64)]

    gen = DummyGen(task_name="dummy")
    dp = DifficultyParams(level="easy", params={"size": 4})
    puzzle = gen.generate(dp, seed=42, num_distractors=3)

    assert isinstance(puzzle, PuzzleInstance)
    assert puzzle.task == "dummy"
    assert puzzle.seed == 42
    assert len(puzzle.distractor_svgs) == 3
    assert len(puzzle.distractor_violations) == 3


def test_base_generator_deterministic():
    """Same seed must produce identical puzzles."""
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class DummyGen(BaseGenerator):
        def _generate_puzzle(self, difficulty, rng):
            val = rng.integers(0, 1000)
            return {"val": int(val)}, {"answer": int(val) * 2}

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return f"<svg>{solution_data['answer']}</svg>"

        def _generate_puzzle_svg(self, puzzle_data):
            return f"<svg>{puzzle_data['val']}</svg>"

        def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return "<svg>d</svg>", violation_type

        def _available_violations(self):
            return ["type_a"]

        def verify(self, puzzle, candidate_png):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="size", min_val=4, max_val=64)]

    gen = DummyGen(task_name="dummy")
    dp = DifficultyParams(level="easy", params={})
    p1 = gen.generate(dp, seed=12345, num_distractors=2)
    p2 = gen.generate(dp, seed=12345, num_distractors=2)
    assert p1.puzzle_svg == p2.puzzle_svg
    assert p1.solution_svg == p2.solution_svg
