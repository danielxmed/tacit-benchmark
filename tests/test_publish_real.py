"""Tests for real puzzle generation path in scripts/publish_hf.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_hf import build_snapshot_structure

# PNG magic bytes
_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _minimal_config() -> dict:
    """Return a tiny config dict (1 task, 1 difficulty, 1 puzzle)."""
    return {
        "version": "0.1.0",
        "seed": 42,
        "distractors_per_puzzle": 2,
        "tasks": {
            "maze": {
                "enabled": True,
                "count_per_difficulty": 1,
                "difficulties": {
                    "easy": {"grid_size": 8, "layers": 1, "portals": 0},
                },
            },
        },
    }


class TestRealGenerationSingleRes:
    """Tests using use_generators=True with a single resolution."""

    @pytest.fixture()
    def snapshot(self, tmp_path):
        config = _minimal_config()
        build_snapshot_structure(
            tmp_path / "out",
            config,
            use_generators=True,
            resolutions=[256],
        )
        return tmp_path / "out"

    def test_real_generation_produces_valid_pngs(self, snapshot):
        # Single resolution → flat layout (no res subdir)
        puzzle_png = snapshot / "task_01_maze" / "easy" / "puzzle_0000.png"
        solution_png = snapshot / "task_01_maze" / "easy" / "solution_0000.png"

        assert puzzle_png.exists()
        assert solution_png.exists()

        puzzle_bytes = puzzle_png.read_bytes()
        assert len(puzzle_bytes) > 100
        assert puzzle_bytes[:8] == _PNG_HEADER

        solution_bytes = solution_png.read_bytes()
        assert len(solution_bytes) > 100
        assert solution_bytes[:8] == _PNG_HEADER

    def test_real_generation_produces_distractors(self, snapshot):
        dist_dir = snapshot / "task_01_maze" / "easy" / "distractors_0000"
        assert dist_dir.is_dir()

        pngs = sorted(dist_dir.glob("distractor_*.png"))
        assert len(pngs) == 2

        for p in pngs:
            data = p.read_bytes()
            assert len(data) > 100
            assert data[:8] == _PNG_HEADER

    def test_real_generation_writes_rich_metadata(self, snapshot):
        meta_path = snapshot / "task_01_maze" / "easy" / "meta_0000.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert "distractor_violations" in meta
        assert "resolutions" in meta
        assert meta["resolutions"] == [256]
        assert isinstance(meta["distractor_violations"], list)
        assert len(meta["distractor_violations"]) == 2


class TestRealGenerationMultiRes:
    """Tests using use_generators=True with multiple resolutions."""

    @pytest.fixture()
    def snapshot(self, tmp_path):
        config = _minimal_config()
        build_snapshot_structure(
            tmp_path / "out",
            config,
            use_generators=True,
            resolutions=[256, 512],
        )
        return tmp_path / "out"

    def test_multi_res_creates_resolution_subdirs(self, snapshot):
        easy = snapshot / "task_01_maze" / "easy"
        assert (easy / "256").is_dir()
        assert (easy / "512").is_dir()

    def test_multi_res_each_subdir_has_puzzle_and_solution(self, snapshot):
        for res in [256, 512]:
            res_dir = snapshot / "task_01_maze" / "easy" / str(res)
            puzzle = res_dir / "puzzle_0000.png"
            solution = res_dir / "solution_0000.png"
            assert puzzle.exists(), f"Missing puzzle at {res}px"
            assert solution.exists(), f"Missing solution at {res}px"

            data = puzzle.read_bytes()
            assert len(data) > 100
            assert data[:8] == _PNG_HEADER

    def test_multi_res_each_subdir_has_distractors(self, snapshot):
        for res in [256, 512]:
            dist_dir = (
                snapshot / "task_01_maze" / "easy" / str(res) / "distractors_0000"
            )
            assert dist_dir.is_dir()
            pngs = sorted(dist_dir.glob("distractor_*.png"))
            assert len(pngs) == 2

    def test_multi_res_higher_res_is_larger(self, snapshot):
        lo = (snapshot / "task_01_maze" / "easy" / "256" / "puzzle_0000.png").stat().st_size
        hi = (snapshot / "task_01_maze" / "easy" / "512" / "puzzle_0000.png").stat().st_size
        assert hi > lo

    def test_multi_res_metadata_at_diff_level(self, snapshot):
        meta_path = snapshot / "task_01_maze" / "easy" / "meta_0000.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["resolutions"] == [256, 512]


class TestPlaceholderAndCallback:
    """Placeholder mode and progress callback tests."""

    def test_placeholder_mode_unchanged(self, tmp_path):
        """Default use_generators=False still produces tiny placeholders."""
        config = _minimal_config()
        out = tmp_path / "placeholder_out"
        build_snapshot_structure(out, config)

        puzzle_png = out / "task_01_maze" / "easy" / "puzzle_0000.png"
        assert puzzle_png.exists()

        data = puzzle_png.read_bytes()
        assert len(data) < 100
        assert data[:8] == _PNG_HEADER

    def test_progress_callback_is_called(self, tmp_path):
        config = _minimal_config()
        calls: list[str] = []
        build_snapshot_structure(
            tmp_path / "cb_out",
            config,
            use_generators=True,
            resolutions=[256],
            progress_callback=calls.append,
        )
        assert len(calls) == 1
        assert "maze" in calls[0]
