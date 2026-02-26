# TACIT Benchmark v0.1.0 — Design Document

**Date:** 2026-02-26
**Author:** Daniel Nobrega Medeiros
**Status:** Approved

---

## 1. Identity & Scope

**Name:** TACIT Benchmark v0.1.0
**Full title:** *TACIT Benchmark: A Programmatic Visual Reasoning Benchmark for Generative and Discriminative Models*
**Author:** Daniel Nobrega Medeiros

**Relationship to prior work:** Standalone benchmark inspired by the same research direction as the TACIT paper (arXiv:2602.07061) — tests implicit, non-linguistic visual reasoning — but not tied to any specific model architecture.

**Core thesis:** Current benchmarks for visual reasoning are either too language-dependent, too easy, or too narrow. TACIT provides a language-minimal, deterministically verifiable, difficulty-parameterized benchmark across 10 tasks spanning 6 reasoning domains, with dual evaluation tracks for both generative and discriminative models.

**What it is NOT:**
- Not a perceptual benchmark (no "what's in this image?")
- Not language-dependent (minimal symbolic notation, no natural language clues)
- Not a toy (research-grade, difficulty-scaled, statistically meaningful sample sizes)

**Documentation languages:** English (primary) + Chinese (full translation)

**Scale for v0.1.0:** 8 categories, 10 tasks, ~100-200 puzzles per task per difficulty level. Architecture supports scaling to 50,000+ per task via config change.

**Distribution:**
- **Source of truth:** GitHub repository (danielxmed) — generators, evaluation harness, documentation
- **Frozen snapshot:** HuggingFace (`tylerxdurden/TACIT-benchmark`) — versioned, checksummed, citable
- **Publish pipeline:** CLI command (`tacit publish`) generates full suite and pushes to HF

**Formal citation:**
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

## 2. Evaluation Architecture — Dual-Track

**Ground truth:** Always an image (SVG source, PNG rasterized).

### Track 1 — Generative
- **Input:** puzzle image
- **Output:** model produces a solution image
- **Evaluation:** programmatic structural verification (not pixel matching)
- **Target:** image-to-image models, strong multimodal generative models
- **Pipeline:** model output → structural parser → `generator.verify()` → pass/fail

### Track 2 — Discriminative
- **Input:** puzzle image + N candidate solution images
- **Output:** model selects correct index
- **Evaluation:** exact index match
- **Target:** vision-capable LLMs, any model that can process images

### Distractor System
- Every generator produces: puzzle + solution + N distractors
- Distractors are near-miss solutions — structurally plausible, violating exactly one constraint
- Distractor difficulty is a first-class parameter:
  - Easy distractors: obvious violations
  - Hard distractors: subtle single-constraint violations
- Track 1 ignores distractors; Track 2 uses them
- Distractor violation type is recorded in metadata for analysis

### Cross-Track Signal
The gap between Track 1 and Track 2 performance on the same model measures generative-vs-discriminative reasoning capability — itself a research contribution.

### Verification Contract
Each generator exports a `verify(puzzle, candidate) -> bool` function. Track 1 calls it on the model's output (after structural parsing). Track 2 uses it during generation to validate ground truth and confirm distractors are incorrect.

---

## 3. The 10 Tasks

### Spatial / Pathfinding

**Task 1 — Multi-layer Mazes**
- **Puzzle:** 2D maze with L layers connected by portal pairs. Single entry, single exit.
- **Solution:** the valid path drawn on the maze, crossing layers via portals.
- **Difficulty axis:** grid size (N×N), layer count (L), portal count, dead-end density.
- **Verification:** path connectivity check from start to end, respecting walls and portal links.

### Pattern / Sequence

**Task 2 — Raven's Progressive Matrices**
- **Puzzle:** 3×3 grid of visual tiles with one missing (bottom-right). Tiles follow transformation rules across rows/columns (shape, color, rotation, count).
- **Solution:** the correct missing tile.
- **Difficulty axis:** number of simultaneous transformation rules, rule complexity (additive vs. compositional).
- **Verification:** extracted tile attributes match the unique valid completion.

**Task 3 — Cellular Automata Forward Prediction**
- **Puzzle:** grid at state T + visual encoding of transition rules.
- **Solution:** grid at state T+k.
- **Difficulty axis:** grid size, rule complexity, step count k.
- **Verification:** deterministic simulation of k steps.

**Task 4 — Cellular Automata Inverse Inference**
- **Puzzle:** grid at state T and grid at state T+k.
- **Solution:** the transition rule that connects them (rendered visually as a rule table/diagram).
- **Difficulty axis:** rule space size, k distance, grid ambiguity.
- **Verification:** apply inferred rule to state T for k steps, must produce state T+k exactly.

### Logical Constraint

**Task 5 — Visual Logic Grids**
- **Puzzle:** constraint grid using only symbols/colors as clues. No natural language.
- **Solution:** completed grid satisfying all constraints.
- **Difficulty axis:** grid dimensions, constraint count, constraint type variety.
- **Verification:** all constraints satisfied, solution unique.

### Graph / Connectivity

**Task 6 — Planar Graph k-Coloring**
- **Puzzle:** planar graph drawing with k colors specified.
- **Solution:** same graph with valid k-coloring applied.
- **Difficulty axis:** node count, edge density, k value (closer to chromatic number = harder).
- **Verification:** no adjacent nodes share a color, exactly k colors used.

**Task 7 — Graph Isomorphism Detection**
- **Puzzle:** two graph drawings with different layouts.
- **Solution:** binary — isomorphic or not.
- **Difficulty axis:** node count, structural similarity between non-isomorphic pairs, layout distortion.
- **Verification:** canonical form comparison (nauty/bliss algorithm).

### Topology

**Task 8 — Unknot Detection**
- **Puzzle:** 2D knot diagram projection.
- **Solution:** binary — unknot or not.
- **Difficulty axis:** crossing number, Reidemeister complexity (minimum moves to simplify).
- **Verification:** knot invariant computation (Jones polynomial or similar).

### Geometric / Projection

**Task 9 — Orthographic Projection Identification**
- **Puzzle:** 3D solid in isometric view + specified projection axis.
- **Solution:** correct 2D orthographic projection.
- **Difficulty axis:** solid complexity (face count, concavities), axis ambiguity.
- **Verification:** computed projection matches ground truth exactly.

**Task 10 — Isometric Reconstruction**
- **Puzzle:** three orthographic projections (front, top, side).
- **Solution:** correct isometric view of the 3D solid.
- **Difficulty axis:** solid complexity, projection ambiguity (multiple solids can share some projections).
- **Verification:** re-project the 3D solid along all three axes, must match input projections.

### Design Notes
- All 10 tasks: deterministic verification, no natural language, parameterized difficulty.
- Forward/inverse pairs (3↔4, 9↔10) test qualitatively different reasoning on the same domain.
- Binary tasks (7, 8) use visual indicators for Track 1 and candidate pairs for Track 2.

---

## 4. Technical Architecture

### Image Format Strategy
- **Generation:** SVG (vector, lossless, precise)
- **Distribution:** multi-resolution PNG (256, 512, 1024) for HF snapshot
- **SVGs available** in the repo for researchers who want them
- **Rasterization:** cairosvg for high-quality SVG→PNG

### Tech Stack
- **Core:** Python, numpy, scipy, pillow, svgwrite, pyyaml
- **Graphs:** networkx, pynauty (canonical form for task 7)
- **Topology:** pyknotid or snappy (knot invariants for task 8)
- **3D Geometry:** trimesh, numpy-stl (tasks 9-10)
- **Rendering:** cairosvg (SVG→PNG rasterization)
- **Distribution:** huggingface_hub, datasets

### Rendering Layer
Thin shared rendering abstraction that all generators use. Ensures visual consistency across all 10 tasks. Specialized libraries underneath, unified API on top. Not a framework — just enough to enforce consistent style, colors, line weights, and resolution.

### Generator Contract
Every generator implements the same protocol:

```python
class Generator(Protocol):
    def generate(self, difficulty: DifficultyParams, seed: int) -> Puzzle:
        """Produces puzzle + solution + distractors + verification fn."""
        ...

    def verify(self, puzzle: Puzzle, candidate: Image) -> bool:
        """Deterministic structural verification."""
        ...

    def difficulty_axes(self) -> dict[str, Range]:
        """Declares this task's difficulty parameters."""
        ...
```

### Data Format (per puzzle instance)
```json
{
    "task": "maze",
    "difficulty": {"grid_size": 32, "layers": 3, "portals": 5},
    "seed": 42,
    "puzzle_svg": "path/to/puzzle.svg",
    "puzzle_png": {"256": "...", "512": "...", "1024": "..."},
    "solution_svg": "path/to/solution.svg",
    "solution_png": {"256": "...", "512": "...", "1024": "..."},
    "distractors_png": ["distractor_0.png", "...", "distractor_N.png"],
    "distractor_violations": ["wall_breach", "portal_skip", "..."],
    "verification_hash": "sha256:..."
}
```

---

## 5. Project Structure

```
tacit_benchmark_0.1.0/
├── ABOUT_THE_AUTHOR.md
├── LICENSE
├── README.md / README_zh.md
├── pyproject.toml
├── tacit/
│   ├── __init__.py
│   ├── core/
│   │   ├── renderer.py            # Thin SVG/PNG rendering abstraction
│   │   ├── verifier.py            # Base verification interface
│   │   ├── distractor.py          # Distractor generation framework
│   │   ├── types.py               # Puzzle, Solution, DifficultyParams, etc.
│   │   └── parsers/               # Per-task structural image parsers
│   │       ├── base.py
│   │       ├── maze_parser.py
│   │       ├── graph_parser.py
│   │       └── ...
│   ├── generators/                # One module per task
│   │   ├── maze.py                # Task 1: Multi-layer mazes
│   │   ├── raven.py               # Task 2: Raven's Progressive Matrices
│   │   ├── ca_forward.py          # Task 3: CA forward prediction
│   │   ├── ca_inverse.py          # Task 4: CA inverse inference
│   │   ├── logic_grid.py          # Task 5: Visual logic grids
│   │   ├── graph_coloring.py      # Task 6: Planar graph k-coloring
│   │   ├── graph_isomorphism.py   # Task 7: Graph isomorphism detection
│   │   ├── unknot.py              # Task 8: Unknot detection
│   │   ├── ortho_projection.py    # Task 9: Orthographic projection ID
│   │   └── iso_reconstruction.py  # Task 10: Isometric reconstruction
│   ├── evaluation/
│   │   ├── harness.py             # Core orchestration (task-agnostic)
│   │   ├── track1.py              # Generative track: parse → verify
│   │   ├── track2.py              # Discriminative track: index check
│   │   ├── metrics.py             # Scoring & aggregation
│   │   └── report.py              # Result report generation
│   └── cli.py                     # CLI entry point
├── configs/
│   ├── default.yaml               # Dev/test generation
│   └── full_release.yaml          # Full benchmark generation
├── scripts/
│   └── publish_hf.py              # HF publish logic (called by CLI)
├── tests/
├── docs/
│   ├── en/
│   └── zh/
└── data/                          # Gitignored, generated locally
    ├── svg/
    ├── png/
    └── metadata/
```

---

## 6. CLI Interface

```bash
# Generate puzzles for a specific task
tacit generate --task maze --difficulty hard --count 200 --seed 42

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

## 7. HuggingFace Snapshot Structure

```
tylerxdurden/TACIT-benchmark/
├── README.md                    # Dataset card (formal citation w/ Daniel Nobrega Medeiros)
├── README_zh.md
├── metadata.json                # Version, seed, generation params, checksums
├── task_01_maze/
│   ├── easy/ medium/ hard/ expert/
│   │   ├── puzzle_XXXX.png
│   │   ├── solution_XXXX.png
│   │   ├── distractors_XXXX/
│   │   └── meta_XXXX.json
│   └── task_info.json           # Difficulty axes, verification description
├── task_02_raven/
│   └── ...
└── ...
```

---

## 8. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language dependency | Minimal (symbols/variables only) | Tests visual reasoning, not language understanding |
| Verification | Deterministic, programmatic | No ambiguity, fully reproducible |
| Difficulty | Adaptive per task type | Respects that "hard" means different things per domain |
| Evaluation | Dual-track (generative + discriminative) | Broadens model coverage, cross-track gap is a research signal |
| Distractors | Near-miss, single-constraint violations | Prevents Track 2 from collapsing into trivial pattern matching |
| Image format | SVG generation → multi-res PNG distribution | Precision in generation, universality in consumption |
| Distribution | GitHub repo + HuggingFace frozen snapshot | Reproducibility (generators) + comparability (snapshot) |
| Scale | ~2,000-5,000 for v0.1.0, scalable to 50k+ | Config-driven generation, no code changes needed to scale |
| Rendering | Thin shared abstraction, specialized libs underneath | Visual consistency without over-engineering |
| Documentation | English + Chinese | International reach from day one |
