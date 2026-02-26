# tests/evaluation/test_track1.py
import pytest
from unittest.mock import MagicMock
from tacit.core.types import VerificationResult


def test_track1_correct_solution():
    from tacit.evaluation.track1 import evaluate_generative

    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(passed=True)

    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_svg="<svg>correct</svg>",
    )
    assert result.passed is True


def test_track1_incorrect_solution():
    from tacit.evaluation.track1 import evaluate_generative

    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(
        passed=False, reason="path disconnected"
    )

    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_svg="<svg>wrong</svg>",
    )
    assert result.passed is False


def test_track1_delegates_to_generator():
    """evaluate_generative must call generator.verify with the puzzle and svg."""
    from tacit.evaluation.track1 import evaluate_generative

    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(passed=True)
    mock_puzzle = MagicMock()
    candidate = "<svg>test</svg>"

    evaluate_generative(
        generator=mock_generator,
        puzzle=mock_puzzle,
        candidate_svg=candidate,
    )
    mock_generator.verify.assert_called_once_with(mock_puzzle, candidate)


def test_track1_returns_verification_result_type():
    """Return type must be VerificationResult."""
    from tacit.evaluation.track1 import evaluate_generative

    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(
        passed=True, reason="ok", details={"score": 1.0}
    )

    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_svg="<svg></svg>",
    )
    assert isinstance(result, VerificationResult)
    assert result.details == {"score": 1.0}


def test_track1_preserves_reason():
    """Reason from generator.verify must be preserved in result."""
    from tacit.evaluation.track1 import evaluate_generative

    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(
        passed=False, reason="coloring invalid"
    )

    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_svg="<svg></svg>",
    )
    assert result.reason == "coloring invalid"
