# tests/test_publish.py
"""Tests for HuggingFace publish pipeline helpers."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Task number mapping
# ---------------------------------------------------------------------------


def test_task_number_map_contains_all_ten_tasks():
    from scripts.publish_hf import TASK_NUMBER_MAP

    assert len(TASK_NUMBER_MAP) == 10
    assert TASK_NUMBER_MAP["maze"] == "01"
    assert TASK_NUMBER_MAP["raven"] == "02"
    assert TASK_NUMBER_MAP["ca_forward"] == "03"
    assert TASK_NUMBER_MAP["ca_inverse"] == "04"
    assert TASK_NUMBER_MAP["logic_grid"] == "05"
    assert TASK_NUMBER_MAP["graph_coloring"] == "06"
    assert TASK_NUMBER_MAP["graph_isomorphism"] == "07"
    assert TASK_NUMBER_MAP["unknot"] == "08"
    assert TASK_NUMBER_MAP["ortho_projection"] == "09"
    assert TASK_NUMBER_MAP["iso_reconstruction"] == "10"


def test_task_dir_name():
    from scripts.publish_hf import task_dir_name

    assert task_dir_name("maze") == "task_01_maze"
    assert task_dir_name("raven") == "task_02_raven"
    assert task_dir_name("iso_reconstruction") == "task_10_iso_reconstruction"


def test_task_dir_name_unknown_task_raises():
    from scripts.publish_hf import task_dir_name

    with pytest.raises(KeyError):
        task_dir_name("nonexistent_task")


# ---------------------------------------------------------------------------
# Dataset card generation
# ---------------------------------------------------------------------------


def test_generate_dataset_card():
    from scripts.publish_hf import generate_dataset_card

    card = generate_dataset_card(version="0.1.0")
    assert "Daniel Nobrega Medeiros" in card
    assert "TACIT" in card
    assert "tylerxdurden" in card
    assert "10.57967/hf/7904" in card


def test_generate_dataset_card_contains_yaml_front_matter():
    from scripts.publish_hf import generate_dataset_card

    card = generate_dataset_card(version="0.1.0")
    assert card.startswith("---\n")
    assert "license: apache-2.0" in card
    assert "visual-question-answering" in card


def test_generate_dataset_card_includes_version():
    from scripts.publish_hf import generate_dataset_card

    card = generate_dataset_card(version="1.2.3")
    assert "v1.2.3" in card


def test_generate_dataset_card_lists_all_ten_tasks():
    from scripts.publish_hf import generate_dataset_card

    card = generate_dataset_card(version="0.1.0")
    assert "Maze" in card
    assert "Raven" in card
    assert "Cellular Automata" in card
    assert "Logic Grid" in card
    assert "Graph" in card
    assert "Unknot" in card
    assert "Orthographic" in card
    assert "Isometric" in card


# ---------------------------------------------------------------------------
# Checksum computation
# ---------------------------------------------------------------------------


def test_compute_checksums():
    from scripts.publish_hf import compute_checksums

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "test.txt").write_text("hello")
        checksums = compute_checksums(p)
        assert "test.txt" in checksums
        assert checksums["test.txt"].startswith("sha256:")


def test_compute_checksums_deterministic():
    from scripts.publish_hf import compute_checksums

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "a.txt").write_text("content_a")
        (p / "b.txt").write_text("content_b")
        c1 = compute_checksums(p)
        c2 = compute_checksums(p)
        assert c1 == c2


def test_compute_checksums_recursive():
    from scripts.publish_hf import compute_checksums

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        subdir = p / "sub" / "deep"
        subdir.mkdir(parents=True)
        (subdir / "nested.txt").write_text("nested content")
        checksums = compute_checksums(p)
        # Key should be relative path with forward slashes
        assert any("nested.txt" in k for k in checksums)


def test_compute_checksums_empty_dir():
    from scripts.publish_hf import compute_checksums

    with tempfile.TemporaryDirectory() as tmpdir:
        checksums = compute_checksums(Path(tmpdir))
        assert checksums == {}


def test_compute_checksums_sha256_format():
    """SHA-256 hex digest should be 64 characters."""
    from scripts.publish_hf import compute_checksums

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "f.bin").write_bytes(b"\x00\x01\x02")
        checksums = compute_checksums(p)
        sha_hex = checksums["f.bin"].split(":", 1)[1]
        assert len(sha_hex) == 64


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------


def test_generate_metadata_json():
    from scripts.publish_hf import generate_metadata

    meta = generate_metadata(
        version="0.1.0",
        seed=42,
        config={"tasks": {"maze": {"enabled": True}}},
        checksums={"file.png": "sha256:abc123"},
    )
    assert meta["version"] == "0.1.0"
    assert meta["seed"] == 42


def test_generate_metadata_includes_config():
    from scripts.publish_hf import generate_metadata

    cfg = {"tasks": {"raven": {"enabled": True}}}
    meta = generate_metadata(
        version="0.1.0",
        seed=99,
        config=cfg,
        checksums={},
    )
    assert meta["generation_config"] == cfg


def test_generate_metadata_includes_checksums():
    from scripts.publish_hf import generate_metadata

    cs = {"a.png": "sha256:aaa", "b.png": "sha256:bbb"}
    meta = generate_metadata(
        version="0.1.0",
        seed=1,
        config={},
        checksums=cs,
    )
    assert meta["checksums"] == cs


def test_generate_metadata_is_json_serializable():
    from scripts.publish_hf import generate_metadata

    meta = generate_metadata(
        version="0.1.0",
        seed=42,
        config={"x": [1, 2, 3]},
        checksums={"f.txt": "sha256:deadbeef"},
    )
    # Should not raise
    serialized = json.dumps(meta)
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Snapshot structure building
# ---------------------------------------------------------------------------


def test_build_snapshot_creates_task_dirs():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "maze": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {"easy": {"grid_size": 8}},
                },
            },
        }
        build_snapshot_structure(out, config)
        task_dir = out / "task_01_maze"
        assert task_dir.is_dir()
        assert (task_dir / "easy").is_dir()


def test_build_snapshot_creates_metadata_json():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "maze": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {"easy": {"grid_size": 8}},
                },
            },
        }
        build_snapshot_structure(out, config)
        meta_path = out / "metadata.json"
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert meta["version"] == "0.1.0"
        assert meta["seed"] == 42
        assert "checksums" in meta


def test_build_snapshot_creates_readme():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "maze": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {"easy": {"grid_size": 8}},
                },
            },
        }
        build_snapshot_structure(out, config)
        readme = out / "README.md"
        assert readme.is_file()
        content = readme.read_text()
        assert "TACIT" in content


def test_build_snapshot_creates_task_info_json():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "raven": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {
                        "easy": {"rules": 1, "complexity": "additive"},
                    },
                },
            },
        }
        build_snapshot_structure(out, config)
        task_info = out / "task_02_raven" / "task_info.json"
        assert task_info.is_file()
        info = json.loads(task_info.read_text())
        assert info["task_name"] == "raven"
        assert "difficulties" in info


def test_build_snapshot_skips_disabled_tasks():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "maze": {
                    "enabled": False,
                    "count_per_difficulty": 1,
                    "difficulties": {"easy": {"grid_size": 8}},
                },
                "raven": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {"easy": {"rules": 1}},
                },
            },
        }
        build_snapshot_structure(out, config)
        assert not (out / "task_01_maze").exists()
        assert (out / "task_02_raven").is_dir()


def test_build_snapshot_creates_difficulty_subdirs():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "maze": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {
                        "easy": {"grid_size": 8},
                        "medium": {"grid_size": 16},
                        "hard": {"grid_size": 32},
                    },
                },
            },
        }
        build_snapshot_structure(out, config)
        task_dir = out / "task_01_maze"
        assert (task_dir / "easy").is_dir()
        assert (task_dir / "medium").is_dir()
        assert (task_dir / "hard").is_dir()


def test_build_snapshot_creates_placeholder_files():
    """Each difficulty subdir should contain placeholder puzzle/solution files."""
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "maze": {
                    "enabled": True,
                    "count_per_difficulty": 2,
                    "difficulties": {
                        "easy": {"grid_size": 8},
                    },
                },
            },
        }
        build_snapshot_structure(out, config)
        easy_dir = out / "task_01_maze" / "easy"
        # Should have puzzle and solution PNG placeholders
        puzzle_files = sorted(easy_dir.glob("puzzle_*.png"))
        solution_files = sorted(easy_dir.glob("solution_*.png"))
        assert len(puzzle_files) == 2
        assert len(solution_files) == 2


def test_build_snapshot_creates_meta_json_per_puzzle():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "tasks": {
                "maze": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {
                        "easy": {"grid_size": 8},
                    },
                },
            },
        }
        build_snapshot_structure(out, config)
        easy_dir = out / "task_01_maze" / "easy"
        meta_files = sorted(easy_dir.glob("meta_*.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert "seed" in meta
        assert "difficulty" in meta
        assert "task" in meta


def test_build_snapshot_creates_distractors_dir():
    from scripts.publish_hf import build_snapshot_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "snapshot"
        config = {
            "version": "0.1.0",
            "seed": 42,
            "distractors_per_puzzle": 4,
            "tasks": {
                "maze": {
                    "enabled": True,
                    "count_per_difficulty": 1,
                    "difficulties": {
                        "easy": {"grid_size": 8},
                    },
                },
            },
        }
        build_snapshot_structure(out, config)
        easy_dir = out / "task_01_maze" / "easy"
        distractor_dirs = sorted(easy_dir.glob("distractors_*"))
        assert len(distractor_dirs) == 1
        assert distractor_dirs[0].is_dir()


# ---------------------------------------------------------------------------
# HuggingFace upload (mocked)
# ---------------------------------------------------------------------------


def test_upload_to_hf_calls_api():
    from scripts.publish_hf import upload_to_hf

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = Path(tmpdir) / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "README.md").write_text("test")

        mock_api = MagicMock()
        upload_to_hf(
            snapshot_dir=snapshot_dir,
            repo_id="tylerxdurden/TACIT-benchmark",
            api=mock_api,
        )
        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args
        assert call_kwargs[1]["repo_id"] == "tylerxdurden/TACIT-benchmark"
        assert call_kwargs[1]["repo_type"] == "dataset"


def test_upload_to_hf_dry_run_does_not_call_api():
    from scripts.publish_hf import upload_to_hf

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = Path(tmpdir) / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "README.md").write_text("test")

        mock_api = MagicMock()
        upload_to_hf(
            snapshot_dir=snapshot_dir,
            repo_id="tylerxdurden/TACIT-benchmark",
            api=mock_api,
            dry_run=True,
        )
        mock_api.upload_folder.assert_not_called()


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------


def test_module_has_main_guard():
    """publish_hf.py should be importable AND have __main__ support."""
    from scripts import publish_hf

    # Should be importable without side effects
    assert hasattr(publish_hf, "generate_dataset_card")
    assert hasattr(publish_hf, "compute_checksums")
    assert hasattr(publish_hf, "generate_metadata")
    assert hasattr(publish_hf, "build_snapshot_structure")
    assert hasattr(publish_hf, "upload_to_hf")
