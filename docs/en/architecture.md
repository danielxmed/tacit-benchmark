# TACIT Benchmark v0.1.0 -- Technical Architecture

This document provides a concise overview of the TACIT Benchmark system architecture, covering the generator protocol, rendering layer, verification contract, distractor system, evaluation harness, CLI, and HuggingFace publish pipeline.

For detailed task specifications, see [Task Specifications](task-specifications.md). For evaluation procedures, see [Evaluation Guide](evaluation-guide.md).

---

## Table of Contents

1. [Generator Protocol and Base Class](#1-generator-protocol-and-base-class)
2. [Rendering Layer](#2-rendering-layer)
3. [Verification Contract](#3-verification-contract)
4. [Distractor System](#4-distractor-system)
5. [Evaluation Harness](#5-evaluation-harness)
6. [CLI Commands](#6-cli-commands)
7. [HuggingFace Publish Pipeline](#7-huggingface-publish-pipeline)
8. [Project Layout](#8-project-layout)

---

## 1. Generator Protocol and Base Class

Every task in TACIT is implemented as a generator that conforms to a common protocol and extends a shared base class.

### Protocol (`tacit/core/types.py`)

```python
@runtime_checkable
class GeneratorProtocol(Protocol):
    def generate(self, difficulty: DifficultyParams, seed: int) -> PuzzleInstance: ...
    def verify(self, puzzle: PuzzleInstance, candidate_png: bytes) -> VerificationResult: ...
    def difficulty_axes(self) -> list[DifficultyRange]: ...
```

### Base class (`tacit/generators/base.py`)

`BaseGenerator` is an abstract base class that implements the `generate()` template method and delegates task-specific logic to abstract hooks:

| Hook Method | Responsibility |
|-------------|---------------|
| `_generate_puzzle(difficulty, rng)` | Create puzzle data and solution data structures |
| `_generate_puzzle_svg(puzzle_data)` | Render puzzle to SVG |
| `_generate_solution_svg(puzzle_data, solution_data)` | Render solution to SVG |
| `_generate_distractor(puzzle_data, solution_data, violation_type, rng)` | Create one near-miss distractor |
| `_available_violations()` | List supported violation types |
| `verify(puzzle, candidate_png)` | CV-based verification of a candidate PNG |
| `difficulty_axes()` | Declare difficulty parameters and ranges |

The `generate()` method uses separate RNG streams for puzzle generation (`seed`) and distractor generation (`seed + 2^31`), ensuring puzzle determinism is independent of distractor count.

### Core data types (`tacit/core/types.py`)

| Type | Purpose |
|------|---------|
| `DifficultyParams` | Immutable pair of `level: str` and `params: dict[str, Any]` |
| `DifficultyRange` | Declares one difficulty axis with name, min, max, step, and description |
| `VerificationResult` | Immutable result with `passed: bool`, `reason: str`, `details: dict` |
| `PuzzleInstance` | Complete puzzle with SVGs, distractors, violations, and metadata |

### Generator implementations (`tacit/generators/`)

Each of the 10 tasks has its own module:

| Module | Class | Task |
|--------|-------|------|
| `maze.py` | `MazeGenerator` | Multi-layer mazes |
| `raven.py` | `RavenGenerator` | Raven's progressive matrices |
| `ca_forward.py` | `CAForwardGenerator` | CA forward prediction |
| `ca_inverse.py` | `CAInverseGenerator` | CA inverse inference |
| `logic_grid.py` | `LogicGridGenerator` | Visual logic grids |
| `graph_coloring.py` | `GraphColoringGenerator` | Planar graph k-coloring |
| `graph_isomorphism.py` | `GraphIsomorphismGenerator` | Graph isomorphism detection |
| `unknot.py` | `UnknotGenerator` | Unknot detection |
| `ortho_projection.py` | `OrthoProjectionGenerator` | Orthographic projection ID |
| `iso_reconstruction.py` | `IsoReconstructionGenerator` | Isometric reconstruction |

Shared utilities are in `_ca_common.py` (cellular automata simulation and rendering) and `_geometry_common.py` (voxel model generation, projection, and isometric rendering).

### Generator Registry (`tacit/generators/registry.py`)

The registry module provides a centralized mapping from task names to generator classes, used by the CLI and publish pipeline:

```python
from tacit.generators.registry import get_generator

gen = get_generator("maze")  # Returns a MazeGenerator instance
```

`GENERATOR_CLASSES` is a dict of `{task_name: (module_path, class_name)}` tuples. Generators are lazy-imported on first use to keep startup fast.

---

## 2. Rendering Layer

**Source:** `tacit/core/renderer.py`

The rendering layer is a thin abstraction over `svgwrite` (SVG generation) and `cairosvg` (SVG-to-PNG rasterization). It enforces visual consistency across all 10 tasks through a shared style dictionary:

```python
STYLE = {
    "background": "#FFFFFF",
    "line_width": 2,
    "line_color": "#222222",
    "grid_color": "#CCCCCC",
    "highlight_color": "#FF4444",
    "solution_color": "#2266FF",
    "font_family": "monospace",
    "font_size": 14,
    "colors": [10-color palette],
}
```

### API

| Function | Purpose |
|----------|---------|
| `create_canvas(width, height)` | Create an SVG Drawing with standard background |
| `draw_rect(...)`, `draw_circle(...)`, `draw_line(...)`, `draw_path(...)`, `draw_text(...)` | Primitive drawing operations using shared style defaults |
| `svg_to_string(canvas)` | Convert canvas to SVG string |
| `svg_to_png(canvas, width)` | Rasterize SVG to PNG bytes at specified width |
| `svg_to_png_multi(canvas, widths)` | Rasterize to multiple resolutions |
| `save_svg(canvas, path)`, `save_png(canvas, path, width)` | File output |

Generators call these primitives rather than using `svgwrite` directly (with the exception of `graph_isomorphism.py`, which uses `svgwrite` directly for its side-by-side dual-graph layout).

### Rasterization Pipeline

```
Generator  -->  SVG Drawing  -->  svg_to_string()  -->  SVG string (generation source of truth)
                                          |
                                  svg_to_png(width) -->  PNG bytes (verification + distribution)
```

SVG remains the source of truth for puzzle generation. However, Track 1 verification operates on PNG images: generators rasterize solutions to PNG internally, and model candidates are submitted as PNG files. PNGs are rasterized at configurable resolutions (release default: 512, 1024, 2048 px) for distribution on HuggingFace. When multiple resolutions are configured, each difficulty directory contains resolution sub-directories (e.g., `easy/512/`, `easy/1024/`, `easy/2048/`).

---

## 3. Verification Contract

**Source:** `tacit/core/verifier.py`

The verification contract requires each task to implement two capabilities:

```python
class BaseVerifier(ABC):
    def verify(self, puzzle: PuzzleInstance, candidate_png: bytes) -> VerificationResult: ...
    def extract_structure(self, png_bytes: bytes) -> Any: ...
```

In practice, generators implement `verify()` directly as part of `BaseGenerator` rather than through a separate verifier class. All verification operates on PNG images using computer vision (CV) techniques. The verification approach varies by task:

| Strategy | Tasks | How it works |
|----------|-------|-------------|
| **Pixel sampling** | maze, ca_forward, ca_inverse, logic_grid, graph_coloring, ortho_projection | PNG pixels are sampled at known positions, colors mapped to structural data, then checked against constraints |
| **Color counting** | graph_isomorphism, unknot | Green vs red pixel counts determine binary answer |
| **SSIM comparison** | raven, iso_reconstruction | Structural similarity index between candidate PNG and ground truth PNG |

### Visual Parsers and CV Utilities

Shared CV utilities in `tacit/core/cv_utils.py` provide common primitives for PNG-based verification:

| Function | Purpose |
|----------|---------|
| `png_to_numpy(png_bytes)` | Load PNG bytes into an (H, W, 3) RGB numpy array |
| `sample_color(img, x, y)` | Sample RGB at pixel coordinates |
| `find_closest_palette_color(pixel, palette)` | Map pixel RGB to nearest palette entry |
| `count_color_pixels(img, target_rgb)` | Count pixels within threshold of a target color |
| `compute_ssim(png1, png2)` | Structural similarity index between two PNG images |
| `hex_to_rgb(hex_color)` | Convert hex color string to RGB tuple |
| `color_distance(c1, c2)` | Euclidean distance between two RGB tuples |

Legacy SVG parsers remain in `tacit/core/parsers/` but are no longer used by Track 1 verification. The CA tasks have additional CV parsing logic in `_ca_common.py` (`parse_grid_from_png`, `parse_rule_from_png`). Each generator implements its own CV-based extraction logic inline in its `verify()` method.

---

## 4. Distractor System

**Source:** `tacit/core/distractor.py`

Distractors are near-miss solutions that violate exactly one structural constraint. They exist to support Track 2 (discriminative) evaluation and prevent it from collapsing into trivial pattern matching.

### Design Principles

1. **Single-constraint violation:** Each distractor violates exactly one rule that the correct solution satisfies.
2. **Structural plausibility:** Distractors look like reasonable solutions -- they are not random noise.
3. **Diversity:** Distractors cycle through available violation types to ensure coverage.
4. **Recorded provenance:** The violation type is stored in `PuzzleInstance.distractor_violations` for analysis.

### Base class (`tacit/core/distractor.py`)

```python
class BaseDistractorGenerator(ABC):
    def generate_distractor(self, puzzle_data, solution_data, violation_type, rng) -> tuple[str, str]: ...
    def available_violations(self) -> list[str]: ...
    def generate_set(self, puzzle_data, solution_data, count, rng) -> tuple[list[str], list[str]]: ...
```

In practice, distractor generation is implemented directly in each generator's `_generate_distractor()` and `_available_violations()` methods.

### Generation flow

```
BaseGenerator.generate(difficulty, seed, num_distractors=4)
    |
    +-- _generate_puzzle(difficulty, rng_puzzle)  -->  (puzzle_data, solution_data)
    +-- _generate_puzzle_svg(puzzle_data)         -->  puzzle SVG
    +-- _generate_solution_svg(puzzle_data, sol)  -->  solution SVG
    |
    +-- for i in range(num_distractors):
    |       violation = violations[i % len(violations)]
    |       _generate_distractor(puzzle_data, solution_data, violation, rng_distractor)
    |           -->  (distractor SVG, violation label)
    |
    +-- PuzzleInstance(puzzle_svg, solution_svg, distractor_svgs, distractor_violations, ...)
```

---

## 5. Evaluation Harness

**Source:** `tacit/evaluation/`

The evaluation layer is task-agnostic. It orchestrates evaluation across tasks and tracks without containing task-specific logic.

### Components

| Module | Purpose |
|--------|---------|
| `harness.py` | `EvaluationHarness` class: generator registry, task instantiation, evaluation orchestration |
| `track1.py` | Generative evaluation: delegates to `generator.verify(puzzle, candidate_png)` |
| `track2.py` | Discriminative evaluation: compares `correct_index == selected_index` |
| `metrics.py` | Scoring functions: `compute_accuracy`, `compute_accuracy_by_difficulty`, `compute_accuracy_by_task` |
| `report.py` | JSON report generation |

### Evaluation flow

**Track 1 (Generative):**

```
For each puzzle:
    1. Load puzzle metadata (task, seed, difficulty)
    2. Load model's PNG from model_output/{task}/{difficulty}/{puzzle_id}.png
    3. generator.verify(puzzle, candidate_png) --> VerificationResult
    4. Record passed/failed with reason
```

**Track 2 (Discriminative):**

```
For each entry in results JSON:
    1. Compare correct_index == selected_index
    2. Record correct/incorrect
```

---

## 6. CLI Commands

**Source:** `tacit/cli.py`

The CLI is built on `click` and provides three commands:

### `tacit generate`

Generate puzzle instances for one or more tasks.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--task` | string | None | Single task name |
| `--difficulty` | string | `easy` | Difficulty level |
| `--count` | int | 10 | Number of puzzles |
| `--seed` | int | 42 | Starting random seed |
| `--config` | path | None | YAML config file for batch generation |
| `--output-dir` | path | `data` | Output directory |
| `--distractors` | int | 4 | Number of distractors per puzzle |

Either `--task` or `--config` must be provided.

### `tacit evaluate`

Evaluate model outputs against generated puzzles.

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--track` | choice: generative/discriminative | Yes | Evaluation track |
| `--model-output` | path | Yes | Model output directory or JSON file |
| `--tasks` | string | No (default: all) | Comma-separated task names |
| `--output` | path | No (default: results.json) | Output report path |

### `tacit publish`

Generate a full benchmark suite and push to HuggingFace.

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--config` | path | Yes | Generation config |
| `--hf-repo` | string | Yes | HuggingFace repository |
| `--version-tag` | string | No | Version tag |
| `--output-dir` | path | No (default: `snapshot/`) | Output directory for snapshot |
| `--dry-run` | flag | No | Build snapshot locally without uploading |

---

## 7. HuggingFace Publish Pipeline

**Source:** `scripts/publish_hf.py`

The publish pipeline generates a frozen, checksummed snapshot of the benchmark for distribution on HuggingFace. The pipeline:

1. Reads the generation config (e.g., `configs/release.yaml`).
2. Generates all puzzles for all tasks and difficulty levels.
3. Rasterizes SVGs to multi-resolution PNGs (one SVG generation, multiple rasterizations).
4. Computes SHA-256 checksums for all artifacts.
5. Creates metadata JSON with version, seed, generation parameters, and checksums.
6. Pushes to the specified HuggingFace repository using `huggingface_hub`.

Generation includes retry logic (up to 3 attempts with seed offsets) for robustness.

### HuggingFace snapshot structure

With multiple resolutions (e.g., 512, 1024, 2048), each difficulty directory contains resolution sub-directories:

```
tylerxdurden/TACIT-benchmark/
├── README.md
├── metadata.json
├── task_01_maze/
│   ├── task_info.json
│   ├── easy/
│   │   ├── meta_0000.json
│   │   ├── 512/
│   │   │   ├── puzzle_0000.png
│   │   │   ├── solution_0000.png
│   │   │   └── distractors_0000/
│   │   │       ├── distractor_00.png
│   │   │       └── ...
│   │   ├── 1024/
│   │   │   ├── puzzle_0000.png
│   │   │   └── ...
│   │   └── 2048/
│   │       ├── puzzle_0000.png
│   │       └── ...
│   ├── medium/
│   └── hard/
├── task_02_raven/
│   └── ...
└── task_10_iso_reconstruction/
    └── ...
```

With a single resolution, PNGs are written directly under the difficulty directory (no resolution subdirectory).

---

## 8. Project Layout

```
tacit_benchmark_0.1.0/
├── pyproject.toml                          # Package metadata, dependencies, CLI entry point
├── tacit/
│   ├── __init__.py                         # Package version
│   ├── core/
│   │   ├── types.py                        # DifficultyParams, PuzzleInstance, VerificationResult, GeneratorProtocol
│   │   ├── renderer.py                     # SVG/PNG rendering abstraction
│   │   ├── verifier.py                     # BaseVerifier ABC
│   │   ├── cv_utils.py                     # CV primitives for PNG-based verification
│   │   ├── distractor.py                   # BaseDistractorGenerator ABC
│   │   └── parsers/
│   │       ├── base.py                     # Parser base
│   │       ├── maze_parser.py              # Maze path extraction
│   │       ├── raven_parser.py             # Tile attribute extraction
│   │       ├── graph_parser.py             # Node color extraction
│   │       └── knot_parser.py              # Answer label extraction
│   ├── generators/
│   │   ├── base.py                         # BaseGenerator ABC (template method)
│   │   ├── registry.py                     # Centralized task→generator mapping
│   │   ├── _ca_common.py                   # CA simulation, rule tables, grid/rule PNG parsing
│   │   ├── _geometry_common.py             # Voxel generation, projection, isometric rendering
│   │   ├── maze.py                         # Task 1
│   │   ├── raven.py                        # Task 2
│   │   ├── ca_forward.py                   # Task 3
│   │   ├── ca_inverse.py                   # Task 4
│   │   ├── logic_grid.py                   # Task 5
│   │   ├── graph_coloring.py               # Task 6
│   │   ├── graph_isomorphism.py            # Task 7
│   │   ├── unknot.py                       # Task 8
│   │   ├── ortho_projection.py             # Task 9
│   │   └── iso_reconstruction.py           # Task 10
│   ├── evaluation/
│   │   ├── harness.py                      # EvaluationHarness, generator registry
│   │   ├── track1.py                       # Generative evaluation
│   │   ├── track2.py                       # Discriminative evaluation
│   │   ├── metrics.py                      # Accuracy computation
│   │   └── report.py                       # JSON report generation
│   └── cli.py                              # Click CLI (generate, evaluate, publish)
├── configs/
│   ├── default.yaml                        # Development/test config (10 per difficulty)
│   └── release.yaml                        # Release config (200 per difficulty, multi-resolution)
├── scripts/
│   └── publish_hf.py                       # HuggingFace publish logic
├── tests/                                  # Test suite
├── docs/
│   ├── en/                                 # English documentation
│   └── zh/                                 # Chinese documentation
└── data/                                   # Generated artifacts (gitignored)
```
