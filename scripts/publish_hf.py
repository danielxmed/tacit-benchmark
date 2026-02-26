#!/usr/bin/env python3
"""HuggingFace publish pipeline for TACIT Benchmark.

Provides helper functions for generating and uploading a frozen benchmark
snapshot to HuggingFace Hub. Can be imported as a module (for CLI integration)
or run as a standalone script.

Usage as script:
    python -m scripts.publish_hf --config configs/default.yaml --output snapshot/

Usage as module:
    from scripts.publish_hf import build_snapshot_structure, upload_to_hf
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Task numbering — matches the design doc ordering
# ---------------------------------------------------------------------------

TASK_NUMBER_MAP: dict[str, str] = {
    "maze": "01",
    "raven": "02",
    "ca_forward": "03",
    "ca_inverse": "04",
    "logic_grid": "05",
    "graph_coloring": "06",
    "graph_isomorphism": "07",
    "unknot": "08",
    "ortho_projection": "09",
    "iso_reconstruction": "10",
}


def task_dir_name(task_name: str) -> str:
    """Return the HF directory name for a task, e.g. ``task_01_maze``.

    Raises ``KeyError`` if *task_name* is not a recognised TACIT task.
    """
    number = TASK_NUMBER_MAP[task_name]
    return f"task_{number}_{task_name}"


# ---------------------------------------------------------------------------
# Dataset card generation
# ---------------------------------------------------------------------------


def generate_dataset_card(version: str) -> str:
    """Generate HuggingFace dataset card (README.md)."""
    return f"""---
license: apache-2.0
task_categories:
  - visual-question-answering
  - image-classification
language:
  - en
  - zh
tags:
  - benchmark
  - visual-reasoning
  - puzzle
  - multimodal
size_categories:
  - 1K<n<10K
---

# TACIT Benchmark v{version}

A Programmatic Visual Reasoning Benchmark for Generative and Discriminative Models.

**Author:** Daniel Nobrega Medeiros

## Citation

```bibtex
@misc{{medeiros_2026,
    author       = {{Daniel Nobrega Medeiros}},
    title        = {{TACIT-benchmark}},
    year         = 2026,
    url          = {{https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark}},
    doi          = {{10.57967/hf/7904}},
    publisher    = {{Hugging Face}}
}}
```

## Overview

TACIT provides 10 visual reasoning tasks across 6 domains:
1. Multi-layer Mazes
2. Raven's Progressive Matrices
3. Cellular Automata Forward Prediction
4. Cellular Automata Inverse Inference
5. Visual Logic Grids
6. Planar Graph k-Coloring
7. Graph Isomorphism Detection
8. Unknot Detection
9. Orthographic Projection Identification
10. Isometric Reconstruction

## Evaluation Tracks

- **Track 1 (Generative):** Model produces solution image. Verified programmatically.
- **Track 2 (Discriminative):** Model selects correct solution from N candidates.

See repository for full documentation and evaluation harness.
"""


# ---------------------------------------------------------------------------
# Checksum computation
# ---------------------------------------------------------------------------


def compute_checksums(directory: Path) -> dict[str, str]:
    """Compute SHA-256 checksums for all files in *directory*.

    Returns a mapping of relative file paths (using ``/`` separators) to
    ``sha256:<hex-digest>`` strings.
    """
    checksums: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            rel = str(path.relative_to(directory))
            checksums[rel] = f"sha256:{sha}"
    return checksums


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------


def generate_metadata(
    version: str,
    seed: int,
    config: dict[str, Any],
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Generate ``metadata.json`` contents for the HF snapshot."""
    return {
        "version": version,
        "seed": seed,
        "generation_config": config,
        "checksums": checksums,
    }


# ---------------------------------------------------------------------------
# Snapshot structure building
# ---------------------------------------------------------------------------


def _write_placeholder_png(path: Path) -> None:
    """Write a minimal placeholder PNG file.

    In a full pipeline this would be replaced by actual rendered images
    from the generators. For snapshot structure creation we write a tiny
    valid-ish marker so checksums are meaningful.
    """
    # 1x1 white PNG (smallest valid PNG)
    # Header + IHDR + IDAT + IEND
    _MINIMAL_PNG = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(_MINIMAL_PNG)


def build_snapshot_structure(output_dir: Path, config: dict[str, Any]) -> Path:
    """Build the full HuggingFace snapshot directory structure.

    Creates the directory tree matching the design doc::

        output_dir/
        +-- README.md
        +-- metadata.json
        +-- task_01_maze/
        |   +-- task_info.json
        |   +-- easy/
        |   |   +-- puzzle_0000.png
        |   |   +-- solution_0000.png
        |   |   +-- distractors_0000/
        |   |   +-- meta_0000.json
        |   +-- medium/ ...
        +-- task_02_raven/ ...

    Args:
        output_dir: Root directory for the snapshot.
        config: Generation configuration dict (matching ``configs/*.yaml``).

    Returns:
        The *output_dir* path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    version = config.get("version", "0.1.0")
    seed = config.get("seed", 42)
    distractors_per_puzzle = config.get("distractors_per_puzzle", 4)
    tasks = config.get("tasks", {})

    # --- Write README.md (dataset card) ---
    readme_path = output_dir / "README.md"
    readme_path.write_text(generate_dataset_card(version))

    # --- Create task directories ---
    for task_name, task_cfg in tasks.items():
        if not task_cfg.get("enabled", True):
            continue

        if task_name not in TASK_NUMBER_MAP:
            continue

        tdir = output_dir / task_dir_name(task_name)
        tdir.mkdir(parents=True, exist_ok=True)

        difficulties = task_cfg.get("difficulties", {})
        count_per_diff = task_cfg.get("count_per_difficulty", 1)

        # Write task_info.json
        task_info = {
            "task_name": task_name,
            "task_number": TASK_NUMBER_MAP[task_name],
            "directory": task_dir_name(task_name),
            "difficulties": {
                diff_name: diff_params
                for diff_name, diff_params in difficulties.items()
            },
            "count_per_difficulty": count_per_diff,
        }
        (tdir / "task_info.json").write_text(
            json.dumps(task_info, indent=2) + "\n"
        )

        # Create difficulty subdirectories with placeholder files
        for diff_name, diff_params in difficulties.items():
            diff_dir = tdir / diff_name
            diff_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(count_per_diff):
                puzzle_seed = seed + idx
                suffix = f"{idx:04d}"

                # Puzzle and solution placeholder PNGs
                _write_placeholder_png(diff_dir / f"puzzle_{suffix}.png")
                _write_placeholder_png(diff_dir / f"solution_{suffix}.png")

                # Distractors directory
                dist_dir = diff_dir / f"distractors_{suffix}"
                dist_dir.mkdir(parents=True, exist_ok=True)
                for d_idx in range(distractors_per_puzzle):
                    _write_placeholder_png(
                        dist_dir / f"distractor_{d_idx:02d}.png"
                    )

                # Per-puzzle metadata
                puzzle_meta = {
                    "task": task_name,
                    "difficulty": diff_name,
                    "difficulty_params": diff_params,
                    "seed": puzzle_seed,
                    "puzzle_id": f"{task_name}_{diff_name}_{suffix}",
                    "distractors_count": distractors_per_puzzle,
                }
                (diff_dir / f"meta_{suffix}.json").write_text(
                    json.dumps(puzzle_meta, indent=2) + "\n"
                )

    # --- Compute checksums over the entire tree (excluding metadata.json) ---
    checksums = compute_checksums(output_dir)

    # --- Write metadata.json ---
    metadata = generate_metadata(
        version=version,
        seed=seed,
        config=config,
        checksums=checksums,
    )
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return output_dir


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------


def upload_to_hf(
    snapshot_dir: Path,
    repo_id: str,
    api: Any | None = None,
    dry_run: bool = False,
) -> None:
    """Upload the snapshot directory to HuggingFace Hub.

    Args:
        snapshot_dir: Path to the snapshot directory to upload.
        repo_id: HuggingFace repository identifier
            (e.g. ``tylerxdurden/TACIT-benchmark``).
        api: An ``HfApi`` instance. If *None*, one is created from
            ``huggingface_hub``.
        dry_run: If *True*, skip the actual upload (useful for testing
            the snapshot generation without pushing to HF).
    """
    if dry_run:
        return

    if api is None:  # pragma: no cover
        from huggingface_hub import HfApi

        api = HfApi()

    api.upload_folder(
        folder_path=str(snapshot_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """CLI entry point when run as ``python -m scripts.publish_hf``."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="Generate and optionally publish TACIT benchmark snapshot to HuggingFace."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to generation config YAML (e.g. configs/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="snapshot",
        help="Output directory for the snapshot (default: snapshot/)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo to upload to (e.g. tylerxdurden/TACIT-benchmark). "
        "Omit for dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build snapshot structure without uploading to HuggingFace.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output)

    print(f"Building snapshot structure in {output_dir} ...")
    build_snapshot_structure(output_dir, config)
    print(f"Snapshot structure created at {output_dir}")

    if args.hf_repo and not args.dry_run:
        print(f"Uploading to {args.hf_repo} ...")
        upload_to_hf(snapshot_dir=output_dir, repo_id=args.hf_repo)
        print("Upload complete.")
    else:
        print("Dry-run mode: skipping HuggingFace upload.")


if __name__ == "__main__":
    main()
