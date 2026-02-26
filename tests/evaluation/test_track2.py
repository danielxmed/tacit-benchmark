# tests/evaluation/test_track2.py
import pytest


def test_track2_correct_selection():
    from tacit.evaluation.track2 import evaluate_discriminative
    result = evaluate_discriminative(
        correct_index=2,
        selected_index=2,
    )
    assert result.correct is True


def test_track2_wrong_selection():
    from tacit.evaluation.track2 import evaluate_discriminative
    result = evaluate_discriminative(
        correct_index=2,
        selected_index=0,
    )
    assert result.correct is False
    assert result.selected_index == 0
    assert result.correct_index == 2


def test_track2_result_is_frozen():
    """DiscriminativeResult should be immutable."""
    from tacit.evaluation.track2 import evaluate_discriminative
    result = evaluate_discriminative(correct_index=1, selected_index=1)
    with pytest.raises(AttributeError):
        result.correct = False


def test_track2_boundary_index_zero():
    """Index 0 should work correctly as both correct and selected."""
    from tacit.evaluation.track2 import evaluate_discriminative
    result = evaluate_discriminative(correct_index=0, selected_index=0)
    assert result.correct is True
    assert result.correct_index == 0
    assert result.selected_index == 0


def test_track2_all_indices_compared():
    """Each index should be independently evaluated."""
    from tacit.evaluation.track2 import evaluate_discriminative
    for correct in range(5):
        for selected in range(5):
            result = evaluate_discriminative(
                correct_index=correct,
                selected_index=selected,
            )
            assert result.correct == (correct == selected)
