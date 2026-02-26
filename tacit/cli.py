# tacit/cli.py
"""CLI entry point for TACIT Benchmark."""
from __future__ import annotations

from pathlib import Path

import click
import yaml

import tacit


# ---------------------------------------------------------------------------
# All known task names (matching generator task_name values)
# ---------------------------------------------------------------------------

KNOWN_TASKS: list[str] = [
    "maze",
    "raven",
    "ca_forward",
    "ca_inverse",
    "logic_grid",
    "graph_coloring",
    "graph_isomorphism",
    "unknot",
    "ortho_projection",
    "iso_reconstruction",
]


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=tacit.__version__, prog_name="TACIT Benchmark")
def main() -> None:
    """TACIT Benchmark: A Programmatic Visual Reasoning Benchmark."""
    pass


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@main.command()
@click.option("--task", type=str, default=None, help="Task name (e.g., maze, raven)")
@click.option("--difficulty", type=str, default="easy", help="Difficulty level")
@click.option("--count", type=int, default=10, help="Number of puzzles to generate")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Config YAML file",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data",
    help="Output directory",
)
@click.option("--distractors", type=int, default=4, help="Distractors per puzzle")
def generate(
    task: str | None,
    difficulty: str,
    count: int,
    seed: int,
    config: str | None,
    output_dir: str,
    distractors: int,
) -> None:
    """Generate puzzle instances."""
    if config:
        cfg = yaml.safe_load(Path(config).read_text())
        click.echo(f"Generating from config: {config}")

        from scripts.publish_hf import build_snapshot_structure

        out = Path(output_dir)
        resolutions = sorted(cfg.get("resolutions", [512]))
        build_snapshot_structure(
            out,
            cfg,
            use_generators=True,
            resolutions=resolutions,
            progress_callback=lambda label: click.echo(f"  generated {label}"),
        )
        click.echo(f"Done — output at {out}")
    elif task:
        click.echo(
            f"Generating {count} {task} puzzles at {difficulty} difficulty (seed={seed})"
        )
        # Single-task path kept as echo-only: not all generators handle
        # DifficultyParams(level=..., params={}) with empty params.
    else:
        raise click.UsageError("Provide --task or --config")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--track",
    type=click.Choice(["generative", "discriminative"]),
    required=True,
)
@click.option(
    "--model-output",
    type=click.Path(exists=True),
    required=True,
    help="Model output directory",
)
@click.option(
    "--tasks",
    type=str,
    default="all",
    help="Comma-separated task names or 'all'",
)
@click.option(
    "--output",
    type=click.Path(),
    default="results.json",
    help="Output report path",
)
def evaluate(
    track: str,
    model_output: str,
    tasks: str,
    output: str,
) -> None:
    """Evaluate model outputs against benchmark."""
    click.echo(f"Evaluating track {track} from {model_output}")
    # TODO: load model outputs and run evaluation


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Generation config",
)
@click.option(
    "--hf-repo",
    type=str,
    required=True,
    help="HuggingFace repo (e.g., tylerxdurden/TACIT-benchmark)",
)
@click.option(
    "--version-tag",
    type=str,
    default=None,
    help="Version tag (default: from pyproject.toml)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="snapshot",
    help="Output directory for snapshot (default: snapshot/)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Build snapshot locally without uploading to HuggingFace.",
)
def publish(
    config: str,
    hf_repo: str,
    version_tag: str | None,
    output_dir: str,
    dry_run: bool,
) -> None:
    """Generate and publish frozen snapshot to HuggingFace."""
    from scripts.publish_hf import build_snapshot_structure, upload_to_hf

    cfg = yaml.safe_load(Path(config).read_text())
    out = Path(output_dir)
    resolutions = sorted(cfg.get("resolutions", [512]))

    click.echo(f"Publishing to {hf_repo}")
    click.echo(f"Building snapshot in {out} ...")

    build_snapshot_structure(
        out,
        cfg,
        use_generators=True,
        resolutions=resolutions,
        progress_callback=lambda label: click.echo(f"  generated {label}"),
    )
    click.echo(f"Snapshot created at {out}")

    if not dry_run:
        click.echo(f"Uploading to {hf_repo} ...")
        upload_to_hf(snapshot_dir=out, repo_id=hf_repo)
        click.echo("Upload complete.")
    else:
        click.echo("Dry-run mode: skipping HuggingFace upload.")
