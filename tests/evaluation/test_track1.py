"""Tests for Track 1 (generative) evaluation."""
from unittest.mock import MagicMock

from tacit.core.types import VerificationResult
from tacit.evaluation.track1 import evaluate_generative


def test_track1_correct_solution():
    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(passed=True)
    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_png=b"\x89PNG_correct",
    )
    assert result.passed is True


def test_track1_incorrect_solution():
    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(
        passed=False, reason="wrong"
    )
    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_png=b"\x89PNG_wrong",
    )
    assert result.passed is False


def test_track1_delegates_to_generator():
    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(passed=True)
    mock_puzzle = MagicMock()
    candidate = b"\x89PNG_data"
    evaluate_generative(
        generator=mock_generator,
        puzzle=mock_puzzle,
        candidate_png=candidate,
    )
    mock_generator.verify.assert_called_once_with(mock_puzzle, candidate)


def test_track1_returns_verification_result_type():
    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(
        passed=True, reason="ok", details={"score": 1.0}
    )
    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_png=b"\x89PNG",
    )
    assert isinstance(result, VerificationResult)


def test_track1_preserves_reason():
    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(
        passed=False, reason="specific failure"
    )
    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_png=b"\x89PNG",
    )
    assert result.reason == "specific failure"
