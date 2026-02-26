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
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
  - 10K<n<100K
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


# ---------------------------------------------------------------------------
# Real puzzle generation helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3


def _generate_and_write_puzzle(
    task_name: str,
    diff_name: str,
    diff_params: dict[str, Any],
    idx: int,
    seed: int,
    num_distractors: int,
    diff_dir: Path,
    resolutions: list[int],
) -> dict[str, Any]:
    """Generate one puzzle instance and write PNGs + metadata to *diff_dir*.

    SVGs are generated once and rasterized to each resolution.  When
    *resolutions* has a single entry the PNGs are written directly into
    *diff_dir*; when it has multiple entries each resolution gets a
    sub-directory (``diff_dir/{res}/``).

    Retries up to ``_MAX_RETRIES`` times with seed offsets on failure.

    Returns the per-puzzle metadata dict (suitable for ``meta_XXXX.json``).
    """
    from tacit.core.renderer import svg_string_to_png
    from tacit.core.types import DifficultyParams
    from tacit.generators.registry import get_generator

    suffix = f"{idx:04d}"
    puzzle_seed = seed + idx
    last_error: Exception | None = None
    multi = len(resolutions) > 1

    for attempt in range(_MAX_RETRIES):
        try:
            effective_seed = puzzle_seed + attempt * 1000
            gen = get_generator(task_name)
            difficulty = DifficultyParams(level=diff_name, params=diff_params)
            instance = gen.generate(difficulty, effective_seed, num_distractors)

            # Collect all SVGs that need rasterizing
            all_svgs: list[tuple[str, str]] = [
                (f"puzzle_{suffix}.png", instance.puzzle_svg),
                (f"solution_{suffix}.png", instance.solution_svg),
            ]
            distractor_files: list[tuple[str, str]] = []
            for d_idx, d_svg in enumerate(instance.distractor_svgs):
                distractor_files.append(
                    (f"distractor_{d_idx:02d}.png", d_svg)
                )

            # Rasterize to each resolution
            for res in resolutions:
                if multi:
                    res_dir = diff_dir / str(res)
                    res_dir.mkdir(parents=True, exist_ok=True)
                else:
                    res_dir = diff_dir

                for fname, svg in all_svgs:
                    (res_dir / fname).write_bytes(
                        svg_string_to_png(svg, width=res)
                    )

                dist_dir = res_dir / f"distractors_{suffix}"
                dist_dir.mkdir(parents=True, exist_ok=True)
                for fname, svg in distractor_files:
                    (dist_dir / fname).write_bytes(
                        svg_string_to_png(svg, width=res)
                    )

            # Build rich metadata (written once at diff_dir level)
            puzzle_meta: dict[str, Any] = {
                "task": task_name,
                "difficulty": diff_name,
                "difficulty_params": diff_params,
                "seed": effective_seed,
                "puzzle_id": instance.puzzle_id,
                "distractors_count": num_distractors,
                "distractor_violations": instance.distractor_violations,
                "resolutions": resolutions,
            }
            (diff_dir / f"meta_{suffix}.json").write_text(
                json.dumps(puzzle_meta, indent=2) + "\n"
            )
            return puzzle_meta

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Attempt %d/%d failed for %s/%s/%04d: %s",
                attempt + 1,
                _MAX_RETRIES,
                task_name,
                diff_name,
                idx,
                exc,
            )

    raise RuntimeError(
        f"Failed to generate {task_name}/{diff_name}/{idx:04d} "
        f"after {_MAX_RETRIES} attempts"
    ) from last_error


# ---------------------------------------------------------------------------
# Snapshot structure building
# ---------------------------------------------------------------------------


def build_snapshot_structure(
    output_dir: Path,
    config: dict[str, Any],
    *,
    use_generators: bool = False,
    resolutions: list[int] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Build the full HuggingFace snapshot directory structure.

    When *use_generators* is True and multiple resolutions are given the
    layout nests a resolution sub-directory under each difficulty::

        task_01_maze/easy/512/puzzle_0000.png
        task_01_maze/easy/1024/puzzle_0000.png

    With a single resolution the layout stays flat (no resolution subdir).

    Args:
        output_dir: Root directory for the snapshot.
        config: Generation configuration dict (matching ``configs/*.yaml``).
        use_generators: If *True*, run real generators and rasterize PNGs.
            Defaults to *False* (write tiny placeholder PNGs).
        resolutions: PNG widths in pixels. Defaults to ``[512]``.  When
            *use_generators* is *False* this is ignored.
        progress_callback: Optional callable invoked with a status string
            after each puzzle is generated.

    Returns:
        The *output_dir* path.
    """
    if resolutions is None:
        resolutions = [512]

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

        # Create difficulty subdirectories with puzzle files
        for diff_name, diff_params in difficulties.items():
            diff_dir = tdir / diff_name
            diff_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(count_per_diff):
                if use_generators:
                    _generate_and_write_puzzle(
                        task_name=task_name,
                        diff_name=diff_name,
                        diff_params=diff_params,
                        idx=idx,
                        seed=seed,
                        num_distractors=distractors_per_puzzle,
                        diff_dir=diff_dir,
                        resolutions=resolutions,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            f"{task_name}/{diff_name}/{idx:04d}"
                        )
                else:
                    puzzle_seed = seed + idx
                    suffix = f"{idx:04d}"

                    # Placeholder PNGs
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
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Write placeholder PNGs instead of running real generators.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="PNG width(s) in pixels (default: from config or 512). "
        "Pass multiple values for multi-resolution snapshots.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output)

    use_generators = not args.no_generate

    # Resolve resolutions: CLI flag > config > default
    if args.resolution:
        resolutions = sorted(args.resolution)
    else:
        resolutions = sorted(config.get("resolutions", [512]))

    if use_generators:
        res_str = ", ".join(f"{r}px" for r in resolutions)
        print(f"Generating real puzzles at [{res_str}] into {output_dir} ...")
    else:
        print(f"Building placeholder snapshot in {output_dir} ...")

    puzzle_count = [0]

    def _progress(label: str) -> None:
        puzzle_count[0] += 1
        print(f"  [{puzzle_count[0]:>5d}] {label}")

    build_snapshot_structure(
        output_dir,
        config,
        use_generators=use_generators,
        resolutions=resolutions,
        progress_callback=_progress if use_generators else None,
    )
    print(f"Snapshot created at {output_dir}")

    if args.hf_repo and not args.dry_run:
        print(f"Uploading to {args.hf_repo} ...")
        upload_to_hf(snapshot_dir=output_dir, repo_id=args.hf_repo)
        print("Upload complete.")
    else:
        print("Dry-run mode: skipping HuggingFace upload.")


if __name__ == "__main__":
    main()
