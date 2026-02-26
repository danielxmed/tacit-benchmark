# tests/core/test_types.py
import pytest
from pathlib import Path


def test_difficulty_params_creation():
    from tacit.core.types import DifficultyParams
    dp = DifficultyParams(level="hard", params={"grid_size": 32, "layers": 3})
    assert dp.level == "hard"
    assert dp.params["grid_size"] == 32


def test_puzzle_instance_creation():
    from tacit.core.types import PuzzleInstance, DifficultyParams
    import numpy as np

    difficulty = DifficultyParams(level="easy", params={"grid_size": 8})
    puzzle = PuzzleInstance(
        task="maze",
        puzzle_id="maze_easy_0042",
        seed=42,
        difficulty=difficulty,
        puzzle_svg="<svg></svg>",
        solution_svg="<svg></svg>",
        distractor_svgs=["<svg></svg>", "<svg></svg>"],
        distractor_violations=["wall_breach", "portal_skip"],
        metadata={"layers": 1},
    )
    assert puzzle.task == "maze"
    assert puzzle.puzzle_id == "maze_easy_0042"
    assert len(puzzle.distractor_svgs) == 2
    assert len(puzzle.distractor_violations) == 2


def test_puzzle_instance_validation_mismatched_distractors():
    from tacit.core.types import PuzzleInstance, DifficultyParams

    difficulty = DifficultyParams(level="easy", params={})
    with pytest.raises(ValueError, match="distractor"):
        PuzzleInstance(
            task="maze",
            puzzle_id="maze_easy_0001",
            seed=1,
            difficulty=difficulty,
            puzzle_svg="<svg></svg>",
            solution_svg="<svg></svg>",
            distractor_svgs=["<svg></svg>"],
            distractor_violations=["a", "b"],  # mismatch
            metadata={},
        )


def test_difficulty_range():
    from tacit.core.types import DifficultyRange
    r = DifficultyRange(name="grid_size", min_val=4, max_val=128, step=4)
    assert r.name == "grid_size"
    assert r.min_val == 4


def test_generator_protocol_interface():
    """Verify the Generator protocol is importable and defines the expected methods."""
    from tacit.core.types import GeneratorProtocol
    assert hasattr(GeneratorProtocol, "generate")
    assert hasattr(GeneratorProtocol, "verify")
    assert hasattr(GeneratorProtocol, "difficulty_axes")


def test_verification_result():
    from tacit.core.types import VerificationResult
    vr = VerificationResult(passed=True, details={"path_valid": True})
    assert vr.passed is True
    assert vr.details["path_valid"] is True

    vr_fail = VerificationResult(passed=False, reason="path disconnected")
    assert vr_fail.passed is False
    assert vr_fail.reason == "path disconnected"
