# TACIT Benchmark v0.1.0

**A Programmatic Visual Reasoning Benchmark for Generative and Discriminative Models**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Dataset on HF](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark)

**Author:** Daniel Nobrega Medeiros
([GitHub](https://github.com/danielxmed) |
[Google Scholar](https://scholar.google.com.br/citations?user=D_6AZoEAAAAJ&hl=pt-BR) |
[arXiv](https://arxiv.org/abs/2602.07061))

---

## Overview

TACIT Benchmark is a language-minimal, deterministically verifiable, difficulty-parameterized benchmark for evaluating visual reasoning in AI models. It provides 10 tasks spanning 6 reasoning domains, with dual evaluation tracks for both generative and discriminative models.

Current benchmarks for visual reasoning are either too language-dependent, too easy, or too narrow. TACIT addresses this by using purely visual puzzles with no natural language clues, programmatic verification with no subjective judgment, and configurable difficulty axes per task.

For detailed documentation, see [`docs/en/`](docs/en/).

---

## Tasks

| # | Task | Domain | Description | Difficulty Axes |
|---|------|--------|-------------|-----------------|
| 1 | Multi-layer Mazes | Spatial / Pathfinding | Navigate a 2D maze with portals connecting multiple layers | Grid size, layer count, portal count, dead-end density |
| 2 | Raven's Progressive Matrices | Pattern / Sequence | Complete a 3x3 grid of visual tiles following transformation rules | Rule count, rule complexity |
| 3 | Cellular Automata Forward | Pattern / Sequence | Predict grid state after k steps given transition rules | Grid size, rule complexity, step count |
| 4 | Cellular Automata Inverse | Pattern / Sequence | Infer the transition rule connecting two grid states | Rule space size, step distance, grid ambiguity |
| 5 | Visual Logic Grids | Logical Constraint | Complete a constraint grid using only symbols and colors as clues | Grid dimensions, constraint count, constraint variety |
| 6 | Planar Graph k-Coloring | Graph / Connectivity | Apply a valid k-coloring to a planar graph | Node count, edge density, k value |
| 7 | Graph Isomorphism | Graph / Connectivity | Determine whether two graph drawings are isomorphic | Node count, structural similarity, layout distortion |
| 8 | Unknot Detection | Topology | Determine whether a knot diagram is the unknot | Crossing number, Reidemeister complexity |
| 9 | Orthographic Projection | Geometric / Projection | Identify the correct 2D projection of a 3D solid | Solid complexity, axis ambiguity |
| 10 | Isometric Reconstruction | Geometric / Projection | Reconstruct a 3D solid from three orthographic projections | Solid complexity, projection ambiguity |

All tasks use deterministic programmatic verification, require no natural language, and support parameterized difficulty levels.

---

## Dual-Track Evaluation

TACIT evaluates models along two complementary tracks:

### Track 1 -- Generative

- **Input:** puzzle image
- **Output:** model produces a solution image
- **Evaluation:** programmatic structural verification (not pixel matching)
- **Target:** image-to-image models, strong multimodal generative models
- **Pipeline:** model output --> structural parser --> `generator.verify()` --> pass/fail

### Track 2 -- Discriminative

- **Input:** puzzle image + N candidate solution images (1 correct, N-1 near-miss distractors)
- **Output:** model selects the correct index
- **Evaluation:** exact index match
- **Target:** vision-capable LLMs, any model that can process images

Distractors are structurally plausible near-miss solutions that violate exactly one constraint. The gap between Track 1 and Track 2 performance on the same model measures the generative-vs-discriminative reasoning divide -- itself a research contribution.

---

## Installation

Requires Python 3.11+.

```bash
# Clone the repository
git clone https://github.com/danielxmed/tacit_benchmark_0.1.0.git
cd tacit_benchmark_0.1.0

# Install with all optional dependencies
pip install -e ".[all]"
```

Optional dependency groups:

| Group | Contents |
|-------|----------|
| `publish` | `huggingface_hub`, `datasets` |
| `dev` | `pytest`, `pytest-cov` |
| `topology` | `pyknotid` (knot invariants, Task 8) |
| `graph` | `pynauty` (canonical graph forms, Task 7) |
| `all` | All of the above |

---

## Quick Start

```bash
# Generate 10 easy maze puzzles
tacit generate --task maze --difficulty easy --count 10 --seed 42

# Generate full benchmark suite from config
tacit generate --config configs/default.yaml

# Evaluate a model on Track 1 (generative)
tacit evaluate --track generative --model-output ./results/ --tasks all

# Evaluate a model on Track 2 (discriminative)
tacit evaluate --track discriminative --model-output ./results/ --tasks all

# Publish frozen snapshot to HuggingFace
tacit publish --config configs/full_release.yaml --hf-repo tylerxdurden/TACIT-benchmark
```

---

## CLI Reference

```
tacit generate   Generate puzzle instances for one or more tasks
  --task TEXT         Task name (maze, raven, ca_forward, ca_inverse,
                      logic_grid, graph_coloring, graph_isomorphism,
                      unknot, ortho_projection, iso_reconstruction)
  --difficulty TEXT   Difficulty level (easy, medium, hard, expert)
  --count INT         Number of puzzles to generate
  --seed INT          Random seed for reproducibility
  --config PATH       YAML config file for batch generation

tacit evaluate   Evaluate model outputs against ground truth
  --track TEXT        Evaluation track (generative, discriminative)
  --model-output PATH Directory containing model outputs
  --tasks TEXT        Comma-separated task names, or "all"

tacit publish    Publish a frozen benchmark snapshot to HuggingFace
  --config PATH       YAML config for full release generation
  --hf-repo TEXT      HuggingFace repository (e.g., tylerxdurden/TACIT-benchmark)
```

---

## HuggingFace Dataset

The frozen, versioned, checksummed benchmark snapshot is available at:

[https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark](https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark)

This snapshot is the citable artifact for reproducible comparisons. The GitHub repository contains the generators, evaluation harness, and documentation.

---

## Citation

```bibtex
@misc{medeiros_2026,
    author       = {Daniel Nobrega Medeiros},
    title        = {TACIT-benchmark},
    year         = 2026,
    url          = {https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark},
    doi          = {10.57967/hf/7904},
    publisher    = {Hugging Face}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
