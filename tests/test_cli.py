# tests/test_cli.py
"""Tests for the TACIT Benchmark CLI."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal config YAML for testing."""
    cfg = {
        "version": "0.1.0",
        "seed": 42,
        "output_dir": str(tmp_path / "data"),
        "resolutions": [256],
        "distractors_per_puzzle": 2,
        "tasks": {
            "maze": {
                "enabled": True,
                "count_per_difficulty": 2,
                "difficulties": {
                    "easy": {"grid_size": 8, "layers": 1, "portals": 0},
                },
            },
        },
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(cfg))
    return config_path


# ------------------------------------------------------------------ #
# Help and version tests
# ------------------------------------------------------------------ #


class TestCLIHelp:
    """CLI must expose help text and version for all commands."""

    def test_cli_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "TACIT Benchmark" in result.output

    def test_cli_generate_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--task" in result.output

    def test_cli_evaluate_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--track" in result.output

    def test_cli_publish_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["publish", "--help"])
        assert result.exit_code == 0
        assert "--hf-repo" in result.output

    def test_cli_version(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ------------------------------------------------------------------ #
# Generate command tests
# ------------------------------------------------------------------ #


class TestGenerateCommand:
    """The generate command must accept --task or --config."""

    def test_generate_requires_task_or_config(self, runner):
        """Invoking generate without --task or --config should fail."""
        from tacit.cli import main

        result = runner.invoke(main, ["generate"])
        assert result.exit_code != 0

    def test_generate_with_task_echoes_info(self, runner):
        from tacit.cli import main

        result = runner.invoke(
            main,
            ["generate", "--task", "maze", "--difficulty", "easy", "--count", "5", "--seed", "99"],
        )
        assert result.exit_code == 0
        assert "maze" in result.output
        assert "5" in result.output
        assert "easy" in result.output

    def test_generate_with_config(self, runner, sample_config):
        from tacit.cli import main

        result = runner.invoke(main, ["generate", "--config", str(sample_config)])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_generate_default_values(self, runner):
        """Default difficulty=easy, count=10, seed=42."""
        from tacit.cli import main

        result = runner.invoke(main, ["generate", "--task", "raven"])
        assert result.exit_code == 0
        assert "10" in result.output
        assert "easy" in result.output
        assert "42" in result.output

    def test_generate_task_option_accepts_all_tasks(self, runner):
        """All 10 task names should be accepted."""
        from tacit.cli import main

        tasks = [
            "maze", "raven", "ca_forward", "ca_inverse", "logic_grid",
            "graph_coloring", "graph_isomorphism", "unknot",
            "ortho_projection", "iso_reconstruction",
        ]
        for task in tasks:
            result = runner.invoke(main, ["generate", "--task", task])
            assert result.exit_code == 0, f"Task {task} failed: {result.output}"
            assert task in result.output

    def test_generate_seed_option(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["generate", "--task", "maze", "--seed", "123"])
        assert result.exit_code == 0
        assert "seed=123" in result.output

    def test_generate_distractors_option_in_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["generate", "--help"])
        assert "--distractors" in result.output

    def test_generate_output_dir_option_in_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["generate", "--help"])
        assert "--output-dir" in result.output


# ------------------------------------------------------------------ #
# Evaluate command tests
# ------------------------------------------------------------------ #


class TestEvaluateCommand:
    """The evaluate command must accept --track and --model-output."""

    def test_evaluate_requires_track(self, runner, tmp_path):
        from tacit.cli import main

        result = runner.invoke(main, ["evaluate", "--model-output", str(tmp_path)])
        assert result.exit_code != 0

    def test_evaluate_requires_model_output(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["evaluate", "--track", "generative"])
        assert result.exit_code != 0

    def test_evaluate_track_choices(self, runner, tmp_path):
        """--track should only accept generative or discriminative."""
        from tacit.cli import main

        result = runner.invoke(
            main,
            ["evaluate", "--track", "invalid", "--model-output", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_evaluate_generative_echoes_info(self, runner, tmp_path):
        from tacit.cli import main

        result = runner.invoke(
            main,
            ["evaluate", "--track", "generative", "--model-output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "generative" in result.output.lower()

    def test_evaluate_discriminative_echoes_info(self, runner, tmp_path):
        from tacit.cli import main

        result = runner.invoke(
            main,
            ["evaluate", "--track", "discriminative", "--model-output", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "discriminative" in result.output.lower()

    def test_evaluate_tasks_option(self, runner, tmp_path):
        from tacit.cli import main

        result = runner.invoke(
            main,
            [
                "evaluate",
                "--track", "generative",
                "--model-output", str(tmp_path),
                "--tasks", "maze,raven",
            ],
        )
        assert result.exit_code == 0

    def test_evaluate_output_option_in_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["evaluate", "--help"])
        assert "--output" in result.output


# ------------------------------------------------------------------ #
# Publish command tests
# ------------------------------------------------------------------ #


class TestPublishCommand:
    """The publish command must accept --config and --hf-repo."""

    def test_publish_requires_config(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["publish", "--hf-repo", "user/repo"])
        assert result.exit_code != 0

    def test_publish_requires_hf_repo(self, runner, sample_config):
        from tacit.cli import main

        result = runner.invoke(main, ["publish", "--config", str(sample_config)])
        assert result.exit_code != 0

    def test_publish_echoes_info(self, runner, sample_config):
        from tacit.cli import main

        result = runner.invoke(
            main,
            ["publish", "--config", str(sample_config), "--hf-repo", "user/repo"],
        )
        assert result.exit_code == 0
        assert "user/repo" in result.output

    def test_publish_version_tag_in_help(self, runner):
        from tacit.cli import main

        result = runner.invoke(main, ["publish", "--help"])
        assert "--version-tag" in result.output
