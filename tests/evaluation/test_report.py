# tests/evaluation/test_report.py
import json
import pytest
from pathlib import Path


def test_generate_report_writes_json(tmp_path: Path):
    from tacit.evaluation.report import generate_report
    output = tmp_path / "results" / "report.json"
    data = {"accuracy": 0.75, "total": 100}
    generate_report(data, output)
    assert output.exists()
    loaded = json.loads(output.read_text())
    assert loaded["accuracy"] == 0.75
    assert loaded["total"] == 100


def test_generate_report_creates_parent_dirs(tmp_path: Path):
    from tacit.evaluation.report import generate_report
    output = tmp_path / "deep" / "nested" / "report.json"
    generate_report({"key": "value"}, output)
    assert output.exists()


def test_generate_report_pretty_printed(tmp_path: Path):
    from tacit.evaluation.report import generate_report
    output = tmp_path / "report.json"
    generate_report({"a": 1, "b": 2}, output)
    text = output.read_text()
    # JSON should be indented (pretty-printed)
    assert "\n" in text
    assert "  " in text


def test_generate_report_overwrites_existing(tmp_path: Path):
    from tacit.evaluation.report import generate_report
    output = tmp_path / "report.json"
    generate_report({"version": 1}, output)
    generate_report({"version": 2}, output)
    loaded = json.loads(output.read_text())
    assert loaded["version"] == 2


def test_generate_report_complex_data(tmp_path: Path):
    from tacit.evaluation.report import generate_report
    output = tmp_path / "report.json"
    data = {
        "overall_accuracy": 0.65,
        "by_task": {"maze": 0.8, "raven": 0.5},
        "by_difficulty": {"easy": 0.9, "hard": 0.4},
        "total_puzzles": 200,
    }
    generate_report(data, output)
    loaded = json.loads(output.read_text())
    assert loaded["by_task"]["maze"] == 0.8
    assert loaded["by_difficulty"]["hard"] == 0.4
