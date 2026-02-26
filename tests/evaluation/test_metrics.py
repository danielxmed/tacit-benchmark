# tests/evaluation/test_metrics.py
import pytest


def test_accuracy_computation():
    from tacit.evaluation.metrics import compute_accuracy
    results = [True, True, False, True, False]
    assert compute_accuracy(results) == pytest.approx(0.6)


def test_accuracy_all_correct():
    from tacit.evaluation.metrics import compute_accuracy
    assert compute_accuracy([True, True, True]) == pytest.approx(1.0)


def test_accuracy_all_wrong():
    from tacit.evaluation.metrics import compute_accuracy
    assert compute_accuracy([False, False, False]) == pytest.approx(0.0)


def test_accuracy_empty():
    from tacit.evaluation.metrics import compute_accuracy
    assert compute_accuracy([]) == pytest.approx(0.0)


def test_accuracy_single_correct():
    from tacit.evaluation.metrics import compute_accuracy
    assert compute_accuracy([True]) == pytest.approx(1.0)


def test_accuracy_single_wrong():
    from tacit.evaluation.metrics import compute_accuracy
    assert compute_accuracy([False]) == pytest.approx(0.0)


def test_accuracy_by_difficulty():
    from tacit.evaluation.metrics import compute_accuracy_by_difficulty
    results = [
        {"difficulty": "easy", "correct": True},
        {"difficulty": "easy", "correct": True},
        {"difficulty": "hard", "correct": False},
        {"difficulty": "hard", "correct": True},
    ]
    acc = compute_accuracy_by_difficulty(results)
    assert acc["easy"] == pytest.approx(1.0)
    assert acc["hard"] == pytest.approx(0.5)


def test_accuracy_by_difficulty_single_level():
    from tacit.evaluation.metrics import compute_accuracy_by_difficulty
    results = [
        {"difficulty": "medium", "correct": True},
        {"difficulty": "medium", "correct": False},
    ]
    acc = compute_accuracy_by_difficulty(results)
    assert acc["medium"] == pytest.approx(0.5)
    assert len(acc) == 1


def test_accuracy_by_difficulty_empty():
    from tacit.evaluation.metrics import compute_accuracy_by_difficulty
    acc = compute_accuracy_by_difficulty([])
    assert acc == {}


def test_accuracy_by_task():
    from tacit.evaluation.metrics import compute_accuracy_by_task
    results = [
        {"task": "maze", "correct": True},
        {"task": "maze", "correct": False},
        {"task": "raven", "correct": True},
    ]
    acc = compute_accuracy_by_task(results)
    assert acc["maze"] == pytest.approx(0.5)
    assert acc["raven"] == pytest.approx(1.0)


def test_accuracy_by_task_empty():
    from tacit.evaluation.metrics import compute_accuracy_by_task
    acc = compute_accuracy_by_task([])
    assert acc == {}


def test_accuracy_by_task_many_tasks():
    from tacit.evaluation.metrics import compute_accuracy_by_task
    results = [
        {"task": "maze", "correct": True},
        {"task": "raven", "correct": False},
        {"task": "unknot", "correct": True},
        {"task": "ca_forward", "correct": True},
    ]
    acc = compute_accuracy_by_task(results)
    assert len(acc) == 4
    assert acc["maze"] == pytest.approx(1.0)
    assert acc["raven"] == pytest.approx(0.0)
    assert acc["unknot"] == pytest.approx(1.0)
    assert acc["ca_forward"] == pytest.approx(1.0)
