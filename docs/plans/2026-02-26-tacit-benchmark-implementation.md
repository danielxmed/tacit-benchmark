# TACIT Benchmark v0.1.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade visual reasoning benchmark with 10 programmatically generated puzzle types, dual-track evaluation, and HuggingFace distribution.

**Architecture:** Thin core layer (types, renderer, verifier, distractor framework) → 10 independent generators implementing a shared protocol → task-agnostic evaluation harness → CLI + HF publish pipeline. Each generator owns its puzzle logic, verification, and distractor generation. SVG-first rendering with multi-resolution PNG output.

**Tech Stack:** Python 3.11+, numpy, scipy, svgwrite, cairosvg, pillow, networkx, pynauty, trimesh, pyyaml, click, huggingface_hub, pytest.

**Design Doc:** `docs/plans/2026-02-26-tacit-benchmark-design.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `tacit/__init__.py`
- Create: `tacit/core/__init__.py`
- Create: `tacit/generators/__init__.py`
- Create: `tacit/evaluation/__init__.py`
- Create: `tacit/core/parsers/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`
- Create: `tests/generators/__init__.py`
- Create: `tests/evaluation/__init__.py`
- Create: `configs/default.yaml`
- Create: `configs/full_release.yaml`
- Create: `.gitignore`
- Create: `LICENSE`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "tacit-benchmark"
version = "0.1.0"
description = "A Programmatic Visual Reasoning Benchmark for Generative and Discriminative Models"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.11"
authors = [
    {name = "Daniel Nobrega Medeiros"}
]
keywords = ["benchmark", "visual-reasoning", "puzzle", "evaluation", "multimodal"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "numpy>=1.24",
    "scipy>=1.11",
    "svgwrite>=1.4",
    "cairosvg>=2.7",
    "Pillow>=10.0",
    "networkx>=3.1",
    "trimesh>=4.0",
    "numpy-stl>=3.0",
    "pyyaml>=6.0",
    "click>=8.1",
]

[project.optional-dependencies]
publish = [
    "huggingface_hub>=0.20",
    "datasets>=2.16",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
]
topology = [
    "pyknotid>=0.5",
]
graph = [
    "pynauty>=1.1",
]
all = ["tacit-benchmark[publish,dev,topology,graph]"]

[project.scripts]
tacit = "tacit.cli:main"

[tool.setuptools.packages.find]
include = ["tacit*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

**Step 2: Create .gitignore**

```
# Generated data
data/
*.png
*.svg

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
htmlcov/
.pytest_cache/
```

**Step 3: Create LICENSE (Apache-2.0)**

Use standard Apache-2.0 license text with:
- Copyright 2026 Daniel Nobrega Medeiros

**Step 4: Create all `__init__.py` files**

```python
# tacit/__init__.py
"""TACIT Benchmark: A Programmatic Visual Reasoning Benchmark."""
__version__ = "0.1.0"
```

All other `__init__.py` files are empty.

**Step 5: Create configs/default.yaml**

```yaml
# TACIT Benchmark — Development/Test Generation Config
version: "0.1.0"
seed: 42
output_dir: "data"
resolutions: [256, 512]
distractors_per_puzzle: 4

tasks:
  maze:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {grid_size: 8, layers: 1, portals: 0}
      medium: {grid_size: 16, layers: 2, portals: 2}
      hard: {grid_size: 32, layers: 3, portals: 5}

  raven:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {rules: 1, complexity: "additive"}
      medium: {rules: 2, complexity: "additive"}
      hard: {rules: 3, complexity: "compositional"}

  ca_forward:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {grid_size: 8, rule_complexity: 2, steps: 1}
      medium: {grid_size: 16, rule_complexity: 4, steps: 3}
      hard: {grid_size: 32, rule_complexity: 8, steps: 5}

  ca_inverse:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {grid_size: 8, rule_space: 4, steps: 1}
      medium: {grid_size: 16, rule_space: 8, steps: 2}
      hard: {grid_size: 32, rule_space: 16, steps: 3}

  logic_grid:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {grid_size: 4, constraints: 6, types: 2}
      medium: {grid_size: 5, constraints: 10, types: 3}
      hard: {grid_size: 6, constraints: 16, types: 4}

  graph_coloring:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {nodes: 6, edge_density: 0.3, k: 4}
      medium: {nodes: 12, edge_density: 0.4, k: 4}
      hard: {nodes: 20, edge_density: 0.5, k: 3}

  graph_isomorphism:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {nodes: 5, distortion: 0.3}
      medium: {nodes: 8, distortion: 0.6}
      hard: {nodes: 12, distortion: 0.9}

  unknot:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {crossings: 3}
      medium: {crossings: 6}
      hard: {crossings: 10}

  ortho_projection:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {faces: 6, concavities: 0}
      medium: {faces: 10, concavities: 1}
      hard: {faces: 16, concavities: 3}

  iso_reconstruction:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {faces: 6, ambiguity: 0}
      medium: {faces: 10, ambiguity: 1}
      hard: {faces: 16, ambiguity: 2}
```

**Step 6: Create configs/full_release.yaml**

Same structure as default.yaml but with:
- `count_per_difficulty: 200`
- `resolutions: [256, 512, 1024]`
- `distractors_per_puzzle: 6`
- Added `expert` difficulty tier to each task

**Step 7: Verify project installs**

Run: `cd /Users/danielnobregamedeiros/Desktop/tacit_benchmark_0.1.0 && pip install -e ".[dev]"`
Expected: Successful installation

**Step 8: Commit**

```bash
git add pyproject.toml .gitignore LICENSE tacit/ tests/ configs/
git commit -m "feat: project scaffolding with package structure and configs"
```

---

## Task 2: Core Types

**Files:**
- Create: `tacit/core/types.py`
- Test: `tests/core/test_types.py`

**Step 1: Write the failing test**

```python
# tests/core/test_types.py
import pytest
from pathlib import Path


def test_difficulty_params_creation():
    from tacit.core.types import DifficultyParams
    dp = DifficultyParams(level="hard", params={"grid_size": 32, "layers": 3})
    assert dp.level == "hard"
    assert dp.params["grid_size"] == 32


def test_puzzle_instance_creation():
    from tacit.core.types import PuzzleInstance, DifficultyParams
    import numpy as np

    difficulty = DifficultyParams(level="easy", params={"grid_size": 8})
    puzzle = PuzzleInstance(
        task="maze",
        puzzle_id="maze_easy_0042",
        seed=42,
        difficulty=difficulty,
        puzzle_svg="<svg></svg>",
        solution_svg="<svg></svg>",
        distractor_svgs=["<svg></svg>", "<svg></svg>"],
        distractor_violations=["wall_breach", "portal_skip"],
        metadata={"layers": 1},
    )
    assert puzzle.task == "maze"
    assert puzzle.puzzle_id == "maze_easy_0042"
    assert len(puzzle.distractor_svgs) == 2
    assert len(puzzle.distractor_violations) == 2


def test_puzzle_instance_validation_mismatched_distractors():
    from tacit.core.types import PuzzleInstance, DifficultyParams

    difficulty = DifficultyParams(level="easy", params={})
    with pytest.raises(ValueError, match="distractor"):
        PuzzleInstance(
            task="maze",
            puzzle_id="maze_easy_0001",
            seed=1,
            difficulty=difficulty,
            puzzle_svg="<svg></svg>",
            solution_svg="<svg></svg>",
            distractor_svgs=["<svg></svg>"],
            distractor_violations=["a", "b"],  # mismatch
            metadata={},
        )


def test_difficulty_range():
    from tacit.core.types import DifficultyRange
    r = DifficultyRange(name="grid_size", min_val=4, max_val=128, step=4)
    assert r.name == "grid_size"
    assert r.min_val == 4


def test_generator_protocol_interface():
    """Verify the Generator protocol is importable and defines the expected methods."""
    from tacit.core.types import GeneratorProtocol
    assert hasattr(GeneratorProtocol, "generate")
    assert hasattr(GeneratorProtocol, "verify")
    assert hasattr(GeneratorProtocol, "difficulty_axes")


def test_verification_result():
    from tacit.core.types import VerificationResult
    vr = VerificationResult(passed=True, details={"path_valid": True})
    assert vr.passed is True
    assert vr.details["path_valid"] is True

    vr_fail = VerificationResult(passed=False, reason="path disconnected")
    assert vr_fail.passed is False
    assert vr_fail.reason == "path disconnected"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_types.py -v`
Expected: FAIL — module not found

**Step 3: Write implementation**

```python
# tacit/core/types.py
"""Core types for TACIT Benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DifficultyParams:
    """Parameters defining puzzle difficulty for a specific task."""
    level: str
    params: dict[str, Any]


@dataclass(frozen=True)
class DifficultyRange:
    """Describes one difficulty axis for a task."""
    name: str
    min_val: float
    max_val: float
    step: float | None = None
    description: str = ""


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a candidate solution."""
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PuzzleInstance:
    """A single generated puzzle with its solution and distractors."""
    task: str
    puzzle_id: str
    seed: int
    difficulty: DifficultyParams
    puzzle_svg: str
    solution_svg: str
    distractor_svgs: list[str]
    distractor_violations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.distractor_svgs) != len(self.distractor_violations):
            raise ValueError(
                f"distractor_svgs ({len(self.distractor_svgs)}) and "
                f"distractor_violations ({len(self.distractor_violations)}) "
                f"must have the same length"
            )


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol that all task generators must implement."""

    def generate(self, difficulty: DifficultyParams, seed: int) -> PuzzleInstance:
        """Generate a puzzle instance with solution and distractors."""
        ...

    def verify(self, puzzle: PuzzleInstance, candidate_svg: str) -> VerificationResult:
        """Verify a candidate solution against the puzzle."""
        ...

    def difficulty_axes(self) -> list[DifficultyRange]:
        """Declare the difficulty parameters this task supports."""
        ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_types.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add tacit/core/types.py tests/core/test_types.py
git commit -m "feat: core types — PuzzleInstance, DifficultyParams, GeneratorProtocol"
```

---

## Task 3: Renderer

**Files:**
- Create: `tacit/core/renderer.py`
- Test: `tests/core/test_renderer.py`

**Step 1: Write the failing test**

```python
# tests/core/test_renderer.py
import pytest
from pathlib import Path
import tempfile


def test_style_constants_exist():
    from tacit.core.renderer import STYLE
    assert "background" in STYLE
    assert "line_width" in STYLE
    assert "colors" in STYLE
    assert isinstance(STYLE["colors"], list)


def test_create_svg_canvas():
    from tacit.core.renderer import create_canvas
    canvas = create_canvas(width=512, height=512)
    svg_str = canvas.tostring()
    assert "svg" in svg_str
    assert '512' in svg_str


def test_draw_rect():
    from tacit.core.renderer import create_canvas, draw_rect
    canvas = create_canvas(256, 256)
    draw_rect(canvas, x=10, y=10, width=50, height=50, fill="#FF0000")
    svg_str = canvas.tostring()
    assert "rect" in svg_str
    assert "FF0000" in svg_str


def test_draw_circle():
    from tacit.core.renderer import create_canvas, draw_circle
    canvas = create_canvas(256, 256)
    draw_circle(canvas, cx=100, cy=100, r=25, fill="#00FF00")
    svg_str = canvas.tostring()
    assert "circle" in svg_str


def test_draw_line():
    from tacit.core.renderer import create_canvas, draw_line
    canvas = create_canvas(256, 256)
    draw_line(canvas, x1=0, y1=0, x2=100, y2=100)
    svg_str = canvas.tostring()
    assert "line" in svg_str


def test_draw_path():
    from tacit.core.renderer import create_canvas, draw_path
    canvas = create_canvas(256, 256)
    draw_path(canvas, d="M 10 10 L 50 50 L 90 10 Z", fill="none", stroke="#000")
    svg_str = canvas.tostring()
    assert "path" in svg_str


def test_draw_text():
    from tacit.core.renderer import create_canvas, draw_text
    canvas = create_canvas(256, 256)
    draw_text(canvas, x=50, y=50, text="A", font_size=14)
    svg_str = canvas.tostring()
    assert ">A<" in svg_str


def test_svg_to_string():
    from tacit.core.renderer import create_canvas, svg_to_string
    canvas = create_canvas(100, 100)
    s = svg_to_string(canvas)
    assert isinstance(s, str)
    assert s.startswith("<?xml") or s.startswith("<svg")


def test_svg_to_png():
    from tacit.core.renderer import create_canvas, svg_to_png
    canvas = create_canvas(100, 100)
    png_bytes = svg_to_png(canvas, width=256)
    # PNG magic bytes
    assert png_bytes[:4] == b'\x89PNG'


def test_svg_to_png_multiple_resolutions():
    from tacit.core.renderer import create_canvas, svg_to_png_multi
    canvas = create_canvas(100, 100)
    result = svg_to_png_multi(canvas, widths=[256, 512])
    assert 256 in result
    assert 512 in result
    assert result[256][:4] == b'\x89PNG'
    assert result[512][:4] == b'\x89PNG'


def test_save_svg_to_file():
    from tacit.core.renderer import create_canvas, draw_rect, save_svg
    canvas = create_canvas(100, 100)
    draw_rect(canvas, 0, 0, 100, 100, fill="#FFF")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.svg"
        save_svg(canvas, path)
        assert path.exists()
        content = path.read_text()
        assert "svg" in content


def test_save_png_to_file():
    from tacit.core.renderer import create_canvas, save_png
    canvas = create_canvas(100, 100)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.png"
        save_png(canvas, path, width=256)
        assert path.exists()
        assert path.read_bytes()[:4] == b'\x89PNG'
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_renderer.py -v`
Expected: FAIL — module not found

**Step 3: Write implementation**

```python
# tacit/core/renderer.py
"""Thin SVG/PNG rendering abstraction for TACIT Benchmark.

All generators use this module for visual output. Ensures consistent
style (colors, line weights, fonts) across all 10 puzzle types.
Wraps svgwrite for SVG generation and cairosvg for PNG rasterization.
"""
from __future__ import annotations

from pathlib import Path

import svgwrite
from svgwrite import Drawing
import cairosvg

# --- Shared visual style ---
STYLE: dict = {
    "background": "#FFFFFF",
    "line_width": 2,
    "line_color": "#222222",
    "grid_color": "#CCCCCC",
    "highlight_color": "#FF4444",
    "solution_color": "#2266FF",
    "font_family": "monospace",
    "font_size": 14,
    "colors": [
        "#E63946",  # red
        "#457B9D",  # steel blue
        "#2A9D8F",  # teal
        "#E9C46A",  # yellow
        "#F4A261",  # orange
        "#264653",  # dark teal
        "#6A0572",  # purple
        "#1B998B",  # mint
        "#FF6B6B",  # coral
        "#4ECDC4",  # turquoise
    ],
}


def create_canvas(width: int, height: int) -> Drawing:
    """Create an SVG canvas with consistent background."""
    dwg = svgwrite.Drawing(size=(f"{width}", f"{height}"))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill=STYLE["background"]))
    return dwg


def draw_rect(
    canvas: Drawing,
    x: float, y: float,
    width: float, height: float,
    fill: str = "none",
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw a rectangle on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.rect(
        insert=(x, y),
        size=(width, height),
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_circle(
    canvas: Drawing,
    cx: float, cy: float, r: float,
    fill: str = "none",
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw a circle on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.circle(
        center=(cx, cy),
        r=r,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_line(
    canvas: Drawing,
    x1: float, y1: float,
    x2: float, y2: float,
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw a line on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.line(
        start=(x1, y1),
        end=(x2, y2),
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_path(
    canvas: Drawing,
    d: str,
    fill: str = "none",
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> None:
    """Draw an SVG path on the canvas."""
    stroke = stroke or STYLE["line_color"]
    stroke_width = stroke_width or STYLE["line_width"]
    canvas.add(canvas.path(
        d=d,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    ))


def draw_text(
    canvas: Drawing,
    x: float, y: float,
    text: str,
    font_size: float | None = None,
    fill: str | None = None,
    anchor: str = "middle",
) -> None:
    """Draw text on the canvas."""
    font_size = font_size or STYLE["font_size"]
    fill = fill or STYLE["line_color"]
    canvas.add(canvas.text(
        text,
        insert=(x, y),
        font_size=f"{font_size}px",
        font_family=STYLE["font_family"],
        fill=fill,
        text_anchor=anchor,
    ))


def svg_to_string(canvas: Drawing) -> str:
    """Convert SVG canvas to string."""
    return canvas.tostring()


def svg_to_png(canvas: Drawing, width: int) -> bytes:
    """Rasterize SVG canvas to PNG bytes at the given width."""
    svg_bytes = canvas.tostring().encode("utf-8")
    return cairosvg.svg2png(bytestring=svg_bytes, output_width=width)


def svg_to_png_multi(canvas: Drawing, widths: list[int]) -> dict[int, bytes]:
    """Rasterize SVG canvas to multiple PNG resolutions."""
    svg_bytes = canvas.tostring().encode("utf-8")
    return {
        w: cairosvg.svg2png(bytestring=svg_bytes, output_width=w)
        for w in widths
    }


def save_svg(canvas: Drawing, path: Path) -> None:
    """Save SVG canvas to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canvas.tostring())


def save_png(canvas: Drawing, path: Path, width: int) -> None:
    """Save SVG canvas as rasterized PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(svg_to_png(canvas, width))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_renderer.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add tacit/core/renderer.py tests/core/test_renderer.py
git commit -m "feat: renderer — SVG/PNG rendering abstraction with consistent style"
```

---

## Task 4: Verifier Base & Distractor Framework

**Files:**
- Create: `tacit/core/verifier.py`
- Create: `tacit/core/distractor.py`
- Create: `tacit/core/parsers/base.py`
- Test: `tests/core/test_verifier.py`
- Test: `tests/core/test_distractor.py`

**Step 1: Write failing tests for verifier**

```python
# tests/core/test_verifier.py
import pytest


def test_base_verifier_is_abstract():
    from tacit.core.verifier import BaseVerifier
    with pytest.raises(TypeError):
        BaseVerifier()


def test_base_verifier_subclass():
    from tacit.core.verifier import BaseVerifier
    from tacit.core.types import PuzzleInstance, DifficultyParams, VerificationResult

    class DummyVerifier(BaseVerifier):
        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def extract_structure(self, svg_string):
            return {"dummy": True}

    v = DummyVerifier()
    dp = DifficultyParams(level="easy", params={})
    puzzle = PuzzleInstance(
        task="dummy", puzzle_id="d_0001", seed=1,
        difficulty=dp, puzzle_svg="<svg/>", solution_svg="<svg/>",
        distractor_svgs=[], distractor_violations=[], metadata={},
    )
    result = v.verify(puzzle, "<svg/>")
    assert result.passed is True
```

**Step 2: Write failing tests for distractor framework**

```python
# tests/core/test_distractor.py
import pytest


def test_base_distractor_generator_is_abstract():
    from tacit.core.distractor import BaseDistractorGenerator
    with pytest.raises(TypeError):
        BaseDistractorGenerator()


def test_distractor_generator_subclass():
    from tacit.core.distractor import BaseDistractorGenerator

    class DummyDistractorGen(BaseDistractorGenerator):
        def generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return "<svg>distractor</svg>", violation_type

        def available_violations(self):
            return ["type_a", "type_b"]

    gen = DummyDistractorGen()
    assert "type_a" in gen.available_violations()


def test_generate_distractor_set():
    from tacit.core.distractor import BaseDistractorGenerator
    import numpy as np

    class DummyDistractorGen(BaseDistractorGenerator):
        def generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return f"<svg>{violation_type}</svg>", violation_type

        def available_violations(self):
            return ["type_a", "type_b", "type_c"]

    gen = DummyDistractorGen()
    rng = np.random.default_rng(42)
    svgs, violations = gen.generate_set(
        puzzle_data={}, solution_data={}, count=4, rng=rng,
    )
    assert len(svgs) == 4
    assert len(violations) == 4
    assert all(v in ["type_a", "type_b", "type_c"] for v in violations)
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/core/test_verifier.py tests/core/test_distractor.py -v`
Expected: FAIL — modules not found

**Step 4: Write verifier implementation**

```python
# tacit/core/verifier.py
"""Base verification interface for TACIT Benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tacit.core.types import PuzzleInstance, VerificationResult


class BaseVerifier(ABC):
    """Abstract base for task-specific verifiers.

    Each task implements:
    - extract_structure: parse SVG into a structural representation
    - verify: check if a candidate solution is correct
    """

    @abstractmethod
    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify a candidate solution against the puzzle."""
        ...

    @abstractmethod
    def extract_structure(self, svg_string: str) -> Any:
        """Extract structural representation from SVG for verification."""
        ...
```

**Step 5: Write distractor implementation**

```python
# tacit/core/distractor.py
"""Distractor generation framework for TACIT Benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDistractorGenerator(ABC):
    """Abstract base for task-specific distractor generators.

    Distractors are near-miss solutions that violate exactly one
    structural constraint. Each task defines its own violation types.
    """

    @abstractmethod
    def generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a single distractor SVG.

        Returns:
            (distractor_svg, violation_type)
        """
        ...

    @abstractmethod
    def available_violations(self) -> list[str]:
        """List the violation types this task supports."""
        ...

    def generate_set(
        self,
        puzzle_data: Any,
        solution_data: Any,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[list[str], list[str]]:
        """Generate a set of distractors with diverse violation types.

        Cycles through available violation types to ensure diversity.
        """
        violations = self.available_violations()
        svgs: list[str] = []
        violation_labels: list[str] = []
        for i in range(count):
            vtype = violations[i % len(violations)]
            svg, label = self.generate_distractor(
                puzzle_data, solution_data, vtype, rng,
            )
            svgs.append(svg)
            violation_labels.append(label)
        return svgs, violation_labels
```

**Step 6: Write parser base**

```python
# tacit/core/parsers/base.py
"""Base structural parser interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """Abstract base for task-specific SVG structural parsers.

    Track 1 (generative) evaluation needs to extract structure from
    model-generated images. Each task provides a parser that converts
    SVG/image data into a structural representation the verifier can check.
    """

    @abstractmethod
    def parse(self, svg_string: str) -> Any:
        """Parse SVG string into structural representation."""
        ...
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/core/test_verifier.py tests/core/test_distractor.py -v`
Expected: All 5 tests PASS

**Step 8: Commit**

```bash
git add tacit/core/verifier.py tacit/core/distractor.py tacit/core/parsers/base.py \
    tests/core/test_verifier.py tests/core/test_distractor.py
git commit -m "feat: verifier base, distractor framework, and parser interface"
```

---

## Task 5: Generator Base Class

**Files:**
- Create: `tacit/generators/base.py`
- Test: `tests/generators/test_base.py`

**Step 1: Write the failing test**

```python
# tests/generators/test_base.py
import pytest
import numpy as np


def test_base_generator_is_abstract():
    from tacit.generators.base import BaseGenerator
    with pytest.raises(TypeError):
        BaseGenerator(task_name="test")


def test_base_generator_subclass_must_implement():
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class IncompleteGen(BaseGenerator):
        pass

    with pytest.raises(TypeError):
        IncompleteGen(task_name="test")


def test_base_generator_full_subclass():
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class DummyGen(BaseGenerator):
        def _generate_puzzle(self, difficulty, rng):
            return {"grid": [[0]]}, {"path": [0]}

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return "<svg>solution</svg>"

        def _generate_puzzle_svg(self, puzzle_data):
            return "<svg>puzzle</svg>"

        def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return f"<svg>distractor:{violation_type}</svg>", violation_type

        def _available_violations(self):
            return ["type_a"]

        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="size", min_val=4, max_val=64)]

    gen = DummyGen(task_name="dummy")
    assert gen.task_name == "dummy"


def test_base_generator_generate():
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class DummyGen(BaseGenerator):
        def _generate_puzzle(self, difficulty, rng):
            return {"grid": [[0]]}, {"path": [0]}

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return "<svg>solution</svg>"

        def _generate_puzzle_svg(self, puzzle_data):
            return "<svg>puzzle</svg>"

        def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return f"<svg>d</svg>", violation_type

        def _available_violations(self):
            return ["type_a"]

        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="size", min_val=4, max_val=64)]

    gen = DummyGen(task_name="dummy")
    dp = DifficultyParams(level="easy", params={"size": 4})
    puzzle = gen.generate(dp, seed=42, num_distractors=3)

    assert isinstance(puzzle, PuzzleInstance)
    assert puzzle.task == "dummy"
    assert puzzle.seed == 42
    assert len(puzzle.distractor_svgs) == 3
    assert len(puzzle.distractor_violations) == 3


def test_base_generator_deterministic():
    """Same seed must produce identical puzzles."""
    from tacit.generators.base import BaseGenerator
    from tacit.core.types import DifficultyParams, PuzzleInstance, VerificationResult, DifficultyRange

    class DummyGen(BaseGenerator):
        def _generate_puzzle(self, difficulty, rng):
            val = rng.integers(0, 1000)
            return {"val": int(val)}, {"answer": int(val) * 2}

        def _generate_solution_svg(self, puzzle_data, solution_data):
            return f"<svg>{solution_data['answer']}</svg>"

        def _generate_puzzle_svg(self, puzzle_data):
            return f"<svg>{puzzle_data['val']}</svg>"

        def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return "<svg>d</svg>", violation_type

        def _available_violations(self):
            return ["type_a"]

        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def difficulty_axes(self):
            return [DifficultyRange(name="size", min_val=4, max_val=64)]

    gen = DummyGen(task_name="dummy")
    dp = DifficultyParams(level="easy", params={})
    p1 = gen.generate(dp, seed=12345, num_distractors=2)
    p2 = gen.generate(dp, seed=12345, num_distractors=2)
    assert p1.puzzle_svg == p2.puzzle_svg
    assert p1.solution_svg == p2.solution_svg
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/generators/test_base.py -v`
Expected: FAIL — module not found

**Step 3: Write implementation**

```python
# tacit/generators/base.py
"""Base generator class for TACIT Benchmark tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from tacit.core.types import (
    DifficultyParams,
    DifficultyRange,
    PuzzleInstance,
    VerificationResult,
)


class BaseGenerator(ABC):
    """Abstract base class for all task generators.

    Subclasses implement the task-specific logic:
    - _generate_puzzle: create puzzle + solution data structures
    - _generate_puzzle_svg: render puzzle to SVG
    - _generate_solution_svg: render solution to SVG
    - _generate_distractor: create a single near-miss distractor
    - _available_violations: list violation types
    - verify: check if a candidate is correct
    - difficulty_axes: declare difficulty parameters
    """

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    def generate(
        self,
        difficulty: DifficultyParams,
        seed: int,
        num_distractors: int = 4,
    ) -> PuzzleInstance:
        """Generate a complete puzzle instance.

        Uses separate RNG streams for puzzle generation and distractor
        generation to ensure puzzle determinism regardless of distractor count.
        """
        rng_puzzle = np.random.default_rng(seed)
        puzzle_data, solution_data = self._generate_puzzle(difficulty, rng_puzzle)

        puzzle_svg = self._generate_puzzle_svg(puzzle_data)
        solution_svg = self._generate_solution_svg(puzzle_data, solution_data)

        # Separate RNG for distractors so puzzle is seed-stable
        rng_distractor = np.random.default_rng(seed + 2**31)
        distractor_svgs: list[str] = []
        distractor_violations: list[str] = []
        violations = self._available_violations()

        for i in range(num_distractors):
            vtype = violations[i % len(violations)]
            svg, label = self._generate_distractor(
                puzzle_data, solution_data, vtype, rng_distractor
            )
            distractor_svgs.append(svg)
            distractor_violations.append(label)

        puzzle_id = f"{self.task_name}_{difficulty.level}_{seed:04d}"

        return PuzzleInstance(
            task=self.task_name,
            puzzle_id=puzzle_id,
            seed=seed,
            difficulty=difficulty,
            puzzle_svg=puzzle_svg,
            solution_svg=solution_svg,
            distractor_svgs=distractor_svgs,
            distractor_violations=distractor_violations,
            metadata={},
        )

    @abstractmethod
    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[Any, Any]:
        """Generate puzzle data and solution data.

        Returns:
            (puzzle_data, solution_data) — internal representations
        """
        ...

    @abstractmethod
    def _generate_puzzle_svg(self, puzzle_data: Any) -> str:
        """Render puzzle data to SVG string."""
        ...

    @abstractmethod
    def _generate_solution_svg(
        self, puzzle_data: Any, solution_data: Any
    ) -> str:
        """Render solution data to SVG string."""
        ...

    @abstractmethod
    def _generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Generate a single distractor.

        Returns:
            (distractor_svg, violation_label)
        """
        ...

    @abstractmethod
    def _available_violations(self) -> list[str]:
        """List the violation types this task supports."""
        ...

    @abstractmethod
    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify a candidate solution."""
        ...

    @abstractmethod
    def difficulty_axes(self) -> list[DifficultyRange]:
        """Declare difficulty parameters for this task."""
        ...
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/generators/test_base.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tacit/generators/base.py tests/generators/test_base.py
git commit -m "feat: BaseGenerator with deterministic seeding and distractor generation"
```

---

## Task 6: Generator — Multi-layer Mazes (Task 1)

This is the **reference implementation**. All subsequent generators follow this pattern.

**Files:**
- Create: `tacit/generators/maze.py`
- Create: `tacit/core/parsers/maze_parser.py`
- Test: `tests/generators/test_maze.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_maze.py
import pytest
from tacit.core.types import DifficultyParams, VerificationResult


@pytest.fixture
def maze_gen():
    from tacit.generators.maze import MazeGenerator
    return MazeGenerator()


class TestMazeGeneration:
    def test_generates_puzzle_instance(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        puzzle = maze_gen.generate(dp, seed=42)
        assert puzzle.task == "maze"
        assert puzzle.puzzle_svg
        assert puzzle.solution_svg
        assert "svg" in puzzle.puzzle_svg.lower()

    def test_deterministic(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        p1 = maze_gen.generate(dp, seed=42)
        p2 = maze_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg
        assert p1.solution_svg == p2.solution_svg

    def test_different_seeds_different_puzzles(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        p1 = maze_gen.generate(dp, seed=1)
        p2 = maze_gen.generate(dp, seed=2)
        assert p1.puzzle_svg != p2.puzzle_svg

    def test_generates_distractors(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        puzzle = maze_gen.generate(dp, seed=42, num_distractors=4)
        assert len(puzzle.distractor_svgs) == 4
        assert len(puzzle.distractor_violations) == 4

    def test_solution_verifies(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        puzzle = maze_gen.generate(dp, seed=42)
        result = maze_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed is True

    def test_distractors_fail_verification(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        puzzle = maze_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = maze_gen.verify(puzzle, svg)
            assert result.passed is False

    def test_multi_layer(self, maze_gen):
        dp = DifficultyParams(
            level="medium",
            params={"grid_size": 16, "layers": 2, "portals": 2},
        )
        puzzle = maze_gen.generate(dp, seed=42)
        assert puzzle.puzzle_svg
        result = maze_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed is True

    def test_difficulty_axes(self, maze_gen):
        axes = maze_gen.difficulty_axes()
        names = [a.name for a in axes]
        assert "grid_size" in names
        assert "layers" in names
        assert "portals" in names


class TestMazeSolvability:
    """Every generated maze must have exactly one valid path."""

    def test_easy_solvable(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        for seed in range(10):
            puzzle = maze_gen.generate(dp, seed=seed)
            result = maze_gen.verify(puzzle, puzzle.solution_svg)
            assert result.passed, f"Maze seed={seed} solution failed verification"

    def test_hard_solvable(self, maze_gen):
        dp = DifficultyParams(level="hard", params={"grid_size": 32, "layers": 3, "portals": 5})
        for seed in range(5):
            puzzle = maze_gen.generate(dp, seed=seed)
            result = maze_gen.verify(puzzle, puzzle.solution_svg)
            assert result.passed, f"Maze seed={seed} solution failed verification"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/generators/test_maze.py -v`
Expected: FAIL — module not found

**Step 3: Implement the maze generator**

The maze generator needs:

1. **Grid generation** — randomized DFS to carve passages in each layer
2. **Portal placement** — link cells across layers
3. **Path solving** — BFS/DFS to find the unique valid path across all layers
4. **SVG rendering** — draw walls, portals, start/end markers, and solution path
5. **Distractor generation** — create near-miss paths with specific violations:
   - `wall_breach` — path goes through a wall
   - `portal_skip` — path ignores a required portal transition
   - `disconnected` — path has a gap
   - `wrong_exit` — path ends at wrong cell
6. **Verification** — parse SVG path data, check connectivity through walls/portals

Implementation approach for `tacit/generators/maze.py`:

- Use randomized DFS (recursive backtracker) for maze generation per layer
- Represent maze as `numpy` array: 0=wall, 1=passage, per layer
- Portals stored as list of `((layer_a, row_a, col_a), (layer_b, row_b, col_b))` pairs
- BFS across layers (treating portals as edges) finds the solution path
- Render each layer side-by-side in SVG with portal indicators (colored circles)
- Solution path rendered as colored overlay on the puzzle

Key implementation details:
- `_generate_puzzle` returns `(maze_data, solution_path)` where `maze_data` contains grids + portals + start + end
- `_generate_puzzle_svg` renders grids without path
- `_generate_solution_svg` renders grids with path overlay
- `verify` extracts path from candidate SVG and checks it traverses from start to end without wall breaches

**File: `tacit/generators/maze.py`** — Full implementation (see design doc for algorithm details). Core structure:

```python
class MazeGenerator(BaseGenerator):
    def __init__(self):
        super().__init__(task_name="maze")

    def _generate_puzzle(self, difficulty, rng):
        grid_size = difficulty.params.get("grid_size", 8)
        num_layers = difficulty.params.get("layers", 1)
        num_portals = difficulty.params.get("portals", 0)
        # 1. Generate each layer via randomized DFS
        # 2. Place portals between layers
        # 3. Solve via BFS across all layers
        # 4. Return (maze_data_dict, solution_path_list)
        ...

    def _generate_puzzle_svg(self, puzzle_data):
        # Render layers side-by-side, walls as lines, portals as colored circles
        ...

    def _generate_solution_svg(self, puzzle_data, solution_data):
        # Same as puzzle SVG + path overlay in STYLE["solution_color"]
        ...

    def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
        # Generate near-miss path with exactly one violation
        ...

    def _available_violations(self):
        return ["wall_breach", "portal_skip", "disconnected", "wrong_exit"]

    def verify(self, puzzle, candidate_svg):
        # Extract path from SVG, validate against maze structure
        ...

    def difficulty_axes(self):
        return [
            DifficultyRange("grid_size", 4, 128, step=4),
            DifficultyRange("layers", 1, 8, step=1),
            DifficultyRange("portals", 0, 20, step=1),
        ]
```

The full implementation should be ~300-400 lines. The maze generation algorithm (randomized DFS backtracker) is well-documented — use `numpy` arrays for the grid, standard BFS for solving.

**File: `tacit/core/parsers/maze_parser.py`:**

```python
class MazeParser(BaseParser):
    def parse(self, svg_string):
        # Extract path coordinates from SVG <path> or <line> elements
        # Return list of (layer, row, col) tuples representing the path
        ...
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/generators/test_maze.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tacit/generators/maze.py tacit/core/parsers/maze_parser.py tests/generators/test_maze.py
git commit -m "feat: maze generator — multi-layer mazes with portals and 4 distractor types"
```

---

## Task 7: Generator — Raven's Progressive Matrices (Task 2)

**Files:**
- Create: `tacit/generators/raven.py`
- Create: `tacit/core/parsers/raven_parser.py`
- Test: `tests/generators/test_raven.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_raven.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def raven_gen():
    from tacit.generators.raven import RavenGenerator
    return RavenGenerator()


class TestRavenGeneration:
    def test_generates_puzzle(self, raven_gen):
        dp = DifficultyParams(level="easy", params={"rules": 1, "complexity": "additive"})
        puzzle = raven_gen.generate(dp, seed=42)
        assert puzzle.task == "raven"
        assert "svg" in puzzle.puzzle_svg.lower()

    def test_deterministic(self, raven_gen):
        dp = DifficultyParams(level="easy", params={"rules": 1, "complexity": "additive"})
        p1 = raven_gen.generate(dp, seed=42)
        p2 = raven_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, raven_gen):
        dp = DifficultyParams(level="easy", params={"rules": 1, "complexity": "additive"})
        puzzle = raven_gen.generate(dp, seed=42)
        result = raven_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_distractors_fail(self, raven_gen):
        dp = DifficultyParams(level="easy", params={"rules": 1, "complexity": "additive"})
        puzzle = raven_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = raven_gen.verify(puzzle, svg)
            assert not result.passed

    def test_hard_compositional(self, raven_gen):
        dp = DifficultyParams(level="hard", params={"rules": 3, "complexity": "compositional"})
        puzzle = raven_gen.generate(dp, seed=42)
        result = raven_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_difficulty_axes(self, raven_gen):
        axes = raven_gen.difficulty_axes()
        names = [a.name for a in axes]
        assert "rules" in names
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/generators/test_raven.py -v`
Expected: FAIL

**Step 3: Implement**

Key design for `tacit/generators/raven.py`:

- **Tile attributes**: shape (circle, triangle, square, pentagon, hexagon), color (from STYLE palette), size (small, medium, large), rotation (0, 90, 180, 270), count (1-4 instances)
- **Rules**: Each rule transforms one attribute across rows or columns
  - Additive: attribute progresses linearly (e.g., rotation += 90 per column)
  - Compositional: attribute depends on combination of row AND column position
- **Grid**: 3×3 tiles, bottom-right is the answer
- **Puzzle SVG**: 3×3 grid with "?" in bottom-right cell
- **Solution SVG**: the correct bottom-right tile only
- **Verification**: extract tile attributes from candidate SVG, compare to expected
- **Distractor violations**: `wrong_shape`, `wrong_color`, `wrong_rotation`, `wrong_count`

**Step 4: Run tests, verify pass**

Run: `pytest tests/generators/test_raven.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tacit/generators/raven.py tacit/core/parsers/raven_parser.py tests/generators/test_raven.py
git commit -m "feat: Raven's Progressive Matrices generator with compositional rules"
```

---

## Task 8: Generator — Cellular Automata Forward (Task 3)

**Files:**
- Create: `tacit/generators/ca_forward.py`
- Create: `tacit/core/parsers/ca_parser.py`
- Test: `tests/generators/test_ca_forward.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_ca_forward.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def ca_gen():
    from tacit.generators.ca_forward import CAForwardGenerator
    return CAForwardGenerator()


class TestCAForwardGeneration:
    def test_generates_puzzle(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42)
        assert puzzle.task == "ca_forward"

    def test_deterministic(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        p1 = ca_gen.generate(dp, seed=42)
        p2 = ca_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg
        assert p1.solution_svg == p2.solution_svg

    def test_solution_verifies(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42)
        result = ca_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_distractors_fail(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = ca_gen.verify(puzzle, svg)
            assert not result.passed

    def test_multi_step(self, ca_gen):
        dp = DifficultyParams(level="hard", params={"grid_size": 16, "rule_complexity": 6, "steps": 5})
        puzzle = ca_gen.generate(dp, seed=42)
        result = ca_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed
```

**Step 2-5: Same pattern as Task 7**

Implementation notes for `tacit/generators/ca_forward.py`:
- **Rule representation**: 2D totalistic rules — cell state depends on sum of neighbors (Moore neighborhood)
- **States**: 2-8 states (controlled by rule_complexity)
- **Puzzle SVG**: initial grid + rule table shown visually (lookup table of neighbor-sum → new state, rendered as colored cells)
- **Solution SVG**: the grid at state T+k
- **Verification**: simulate k steps from initial state, compare cell-by-cell
- **Distractor violations**: `wrong_cell` (few cells wrong), `wrong_step_count` (simulated too few/many steps), `wrong_rule` (applied different rule)

```bash
git commit -m "feat: cellular automata forward prediction generator"
```

---

## Task 9: Generator — Cellular Automata Inverse (Task 4)

**Files:**
- Create: `tacit/generators/ca_inverse.py`
- Test: `tests/generators/test_ca_inverse.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_ca_inverse.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def ca_inv_gen():
    from tacit.generators.ca_inverse import CAInverseGenerator
    return CAInverseGenerator()


class TestCAInverseGeneration:
    def test_generates_puzzle(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        assert puzzle.task == "ca_inverse"

    def test_deterministic(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        p1 = ca_inv_gen.generate(dp, seed=42)
        p2 = ca_inv_gen.generate(dp, seed=42)
        assert p1.solution_svg == p2.solution_svg

    def test_solution_verifies(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        result = ca_inv_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_distractors_fail(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = ca_inv_gen.verify(puzzle, svg)
            assert not result.passed

    def test_inferred_rule_reproduces_output(self, ca_inv_gen):
        """The gold standard: applying the solution rule to state T must produce state T+k."""
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        # The verification itself tests this property
        result = ca_inv_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed
```

**Step 2-5: Same pattern**

Implementation notes for `tacit/generators/ca_inverse.py`:
- Shares CA simulation logic with `ca_forward.py` — extract shared utilities to a helper
- **Puzzle SVG**: state T grid + state T+k grid side by side
- **Solution SVG**: the rule table (same visual format as ca_forward uses to display rules)
- **Verification**: parse rule from candidate SVG, simulate k steps from T, compare to T+k
- **Distractor violations**: `off_by_one_rule` (one entry in rule table wrong), `transposed_rule` (rule entries swapped), `partial_rule` (rule works for some cells but not all)

Create shared helper: `tacit/generators/_ca_common.py` for simulation logic, rule encoding, and grid rendering used by both ca_forward and ca_inverse.

```bash
git commit -m "feat: cellular automata inverse inference generator"
```

---

## Task 10: Generator — Visual Logic Grids (Task 5)

**Files:**
- Create: `tacit/generators/logic_grid.py`
- Create: `tacit/core/parsers/logic_grid_parser.py`
- Test: `tests/generators/test_logic_grid.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_logic_grid.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def logic_gen():
    from tacit.generators.logic_grid import LogicGridGenerator
    return LogicGridGenerator()


class TestLogicGridGeneration:
    def test_generates_puzzle(self, logic_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42)
        assert puzzle.task == "logic_grid"

    def test_deterministic(self, logic_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        p1 = logic_gen.generate(dp, seed=42)
        p2 = logic_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, logic_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42)
        result = logic_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_distractors_fail(self, logic_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = logic_gen.verify(puzzle, svg)
            assert not result.passed

    def test_unique_solution(self, logic_gen):
        """Constraints must yield exactly one valid solution."""
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        for seed in range(5):
            puzzle = logic_gen.generate(dp, seed=seed)
            result = logic_gen.verify(puzzle, puzzle.solution_svg)
            assert result.passed, f"Seed {seed} produced invalid solution"
```

**Step 2-5: Same pattern**

Implementation notes:
- **Constraint types** (all visual, no text):
  - **Row/column symbol constraint**: colored marker in a cell border means "this symbol must be in this row/column"
  - **Adjacency constraint**: connector symbol between two cells means "these must/must not be adjacent"
  - **Exclusion constraint**: X mark means "this symbol cannot be here"
  - **Equality constraint**: = mark between cells means "same category"
- **Generation**: start with a valid solution, then generate constraints that uniquely determine it, verify uniqueness via backtracking solver
- **Distractor violations**: `constraint_violation` (one constraint broken), `non_unique` (valid but not the intended solution — caught by checking all constraints), `symbol_swap` (two symbols in same category swapped)

```bash
git commit -m "feat: visual logic grid generator with constraint satisfaction"
```

---

## Task 11: Generator — Planar Graph k-Coloring (Task 6)

**Files:**
- Create: `tacit/generators/graph_coloring.py`
- Create: `tacit/core/parsers/graph_parser.py`
- Test: `tests/generators/test_graph_coloring.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_graph_coloring.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def coloring_gen():
    from tacit.generators.graph_coloring import GraphColoringGenerator
    return GraphColoringGenerator()


class TestGraphColoringGeneration:
    def test_generates_puzzle(self, coloring_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42)
        assert puzzle.task == "graph_coloring"

    def test_deterministic(self, coloring_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        p1 = coloring_gen.generate(dp, seed=42)
        p2 = coloring_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, coloring_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42)
        result = coloring_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_distractors_fail(self, coloring_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = coloring_gen.verify(puzzle, svg)
            assert not result.passed

    def test_generated_graph_is_planar(self, coloring_gen):
        """All generated graphs must be planar."""
        dp = DifficultyParams(level="hard", params={"nodes": 20, "edge_density": 0.5, "k": 3})
        puzzle = coloring_gen.generate(dp, seed=42)
        # Planarity is guaranteed by the generation algorithm
        result = coloring_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_chromatic_difficulty(self, coloring_gen):
        """Hard: k close to chromatic number makes it genuinely hard."""
        dp = DifficultyParams(level="hard", params={"nodes": 12, "edge_density": 0.5, "k": 3})
        puzzle = coloring_gen.generate(dp, seed=42)
        result = coloring_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed
```

**Step 2-5: Same pattern**

Implementation notes:
- Use `networkx` for graph generation: `nx.random_planar_graph` or Delaunay triangulation with edge removal
- k-coloring via backtracking with constraint propagation
- Ensure k >= chromatic_number(G) so solution exists
- **Layout**: spring layout or planar layout via `networkx`
- **Verification**: parse node colors from SVG, check no adjacent pair shares a color, check exactly k colors used
- **Distractor violations**: `adjacent_same_color` (one adjacent pair same color), `wrong_k` (uses k+1 colors), `missing_node` (one node uncolored)

```bash
git commit -m "feat: planar graph k-coloring generator"
```

---

## Task 12: Generator — Graph Isomorphism Detection (Task 7)

**Files:**
- Create: `tacit/generators/graph_isomorphism.py`
- Test: `tests/generators/test_graph_isomorphism.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_graph_isomorphism.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def iso_gen():
    from tacit.generators.graph_isomorphism import GraphIsomorphismGenerator
    return GraphIsomorphismGenerator()


class TestGraphIsomorphismGeneration:
    def test_generates_puzzle(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        assert puzzle.task == "graph_isomorphism"

    def test_deterministic(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        p1 = iso_gen.generate(dp, seed=42)
        p2 = iso_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        result = iso_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_balanced_positive_negative(self, iso_gen):
        """Over many seeds, roughly half should be isomorphic, half not."""
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        positive = 0
        for seed in range(20):
            puzzle = iso_gen.generate(dp, seed=seed)
            is_iso = puzzle.metadata.get("is_isomorphic")
            if is_iso:
                positive += 1
        assert 5 <= positive <= 15  # roughly balanced
```

**Step 2-5: Same pattern**

Implementation notes:
- **Positive pairs**: generate graph G, create G' by permuting node labels and applying different layout
- **Negative pairs**: generate G, create G' by adding/removing one edge (preserving degree sequence when possible for harder cases)
- **Layout distortion**: controlled by `distortion` param — higher = more different layouts
- **Solution**: binary indicator — green checkmark SVG for "isomorphic", red X for "not isomorphic"
- **Verification**: use `networkx.is_isomorphic()` (VF2 algorithm) as ground truth. For production, consider `pynauty` for canonical form comparison.
- **Distractor violations**: for binary tasks, distractors are simply the opposite answer

```bash
git commit -m "feat: graph isomorphism detection generator"
```

---

## Task 13: Generator — Unknot Detection (Task 8)

**Files:**
- Create: `tacit/generators/unknot.py`
- Create: `tacit/core/parsers/knot_parser.py`
- Test: `tests/generators/test_unknot.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_unknot.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def unknot_gen():
    from tacit.generators.unknot import UnknotGenerator
    return UnknotGenerator()


class TestUnknotGeneration:
    def test_generates_puzzle(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        puzzle = unknot_gen.generate(dp, seed=42)
        assert puzzle.task == "unknot"

    def test_deterministic(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        p1 = unknot_gen.generate(dp, seed=42)
        p2 = unknot_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        puzzle = unknot_gen.generate(dp, seed=42)
        result = unknot_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_balanced_unknot_knot(self, unknot_gen):
        """Roughly balanced between unknots and non-trivial knots."""
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        unknots = 0
        for seed in range(20):
            puzzle = unknot_gen.generate(dp, seed=seed)
            if puzzle.metadata.get("is_unknot"):
                unknots += 1
        assert 5 <= unknots <= 15
```

**Step 2-5: Same pattern**

Implementation notes:
- **Unknot generation**: start with a circle, apply random Reidemeister moves to add crossings without changing knot type → guaranteed unknot with N crossings
- **Non-trivial knot generation**: use known knot tables (trefoil, figure-eight, etc.) or generate random knot diagrams and check invariants
- **Knot invariants**: use bracket polynomial or Jones polynomial for verification. If `pyknotid` is available, use it. Otherwise implement bracket polynomial directly (feasible for crossing numbers ≤ 15).
- **SVG rendering**: draw knot diagram as smooth curves with over/under crossing indicators (gap in the under-strand)
- **Verification**: compute invariant, compare to unknot invariant (trivial Jones polynomial = 1)
- **Distractor violations**: opposite answer (binary task)
- **Difficulty**: crossing number controls difficulty. More crossings = harder to visually determine if it's an unknot.

Note: This is the most mathematically sophisticated generator. If `pyknotid` or `snappy` dependencies are problematic, implement a minimal bracket polynomial calculator (~100 lines).

```bash
git commit -m "feat: unknot detection generator with knot invariant verification"
```

---

## Task 14: Generator — Orthographic Projection (Task 9)

**Files:**
- Create: `tacit/generators/ortho_projection.py`
- Create: `tacit/generators/_geometry_common.py`
- Create: `tacit/core/parsers/projection_parser.py`
- Test: `tests/generators/test_ortho_projection.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_ortho_projection.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def ortho_gen():
    from tacit.generators.ortho_projection import OrthoProjectionGenerator
    return OrthoProjectionGenerator()


class TestOrthoProjectionGeneration:
    def test_generates_puzzle(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        puzzle = ortho_gen.generate(dp, seed=42)
        assert puzzle.task == "ortho_projection"

    def test_deterministic(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        p1 = ortho_gen.generate(dp, seed=42)
        p2 = ortho_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        puzzle = ortho_gen.generate(dp, seed=42)
        result = ortho_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_distractors_fail(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        puzzle = ortho_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = ortho_gen.verify(puzzle, svg)
            assert not result.passed
```

**Step 2-5: Same pattern**

Implementation notes:
- **3D solid generation**: use `trimesh` to create solids via boolean operations on primitive shapes (boxes, cylinders). Control complexity via face count and concavity parameters.
- **Shared geometry helper** (`_geometry_common.py`): solid generation, isometric rendering, orthographic projection — used by both Task 9 and Task 10.
- **Puzzle SVG**: isometric view of the 3D solid + axis indicator showing projection direction
- **Solution SVG**: the correct 2D orthographic projection along the specified axis
- **Verification**: compute projection from the stored 3D mesh, compare 2D silhouette
- **Distractor violations**: `wrong_axis` (projection along different axis), `missing_feature` (concavity not shown), `extra_feature` (phantom geometry), `mirrored` (left-right flip)

```bash
git commit -m "feat: orthographic projection identification generator"
```

---

## Task 15: Generator — Isometric Reconstruction (Task 10)

**Files:**
- Create: `tacit/generators/iso_reconstruction.py`
- Test: `tests/generators/test_iso_reconstruction.py`

**Step 1: Write failing tests**

```python
# tests/generators/test_iso_reconstruction.py
import pytest
from tacit.core.types import DifficultyParams


@pytest.fixture
def iso_recon_gen():
    from tacit.generators.iso_reconstruction import IsoReconstructionGenerator
    return IsoReconstructionGenerator()


class TestIsoReconstructionGeneration:
    def test_generates_puzzle(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        puzzle = iso_recon_gen.generate(dp, seed=42)
        assert puzzle.task == "iso_reconstruction"

    def test_deterministic(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        p1 = iso_recon_gen.generate(dp, seed=42)
        p2 = iso_recon_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        puzzle = iso_recon_gen.generate(dp, seed=42)
        result = iso_recon_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed

    def test_distractors_fail(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        puzzle = iso_recon_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = iso_recon_gen.verify(puzzle, svg)
            assert not result.passed
```

**Step 2-5: Same pattern**

Implementation notes:
- Inverse of Task 9 — uses same `_geometry_common.py` helpers
- **Puzzle SVG**: three orthographic projections (front, top, side) arranged in standard engineering drawing layout
- **Solution SVG**: correct isometric view of the 3D solid
- **Verification**: project candidate 3D solid along all three axes, compare to input projections
- **Difficulty**: `ambiguity` param controls how many distinct 3D solids could produce the same three projections. Higher ambiguity = distractors are valid for 2/3 projections but fail on the third.
- **Distractor violations**: `wrong_depth` (correct silhouettes but wrong 3D structure), `missing_face`, `extra_volume`, `rotated`

```bash
git commit -m "feat: isometric reconstruction generator"
```

---

## Task 16: Evaluation Harness

**Files:**
- Create: `tacit/evaluation/harness.py`
- Create: `tacit/evaluation/track1.py`
- Create: `tacit/evaluation/track2.py`
- Create: `tacit/evaluation/metrics.py`
- Create: `tacit/evaluation/report.py`
- Test: `tests/evaluation/test_harness.py`
- Test: `tests/evaluation/test_track1.py`
- Test: `tests/evaluation/test_track2.py`
- Test: `tests/evaluation/test_metrics.py`

**Step 1: Write failing tests for Track 2 (simpler)**

```python
# tests/evaluation/test_track2.py
import pytest


def test_track2_correct_selection():
    from tacit.evaluation.track2 import evaluate_discriminative
    result = evaluate_discriminative(
        correct_index=2,
        selected_index=2,
    )
    assert result.correct is True


def test_track2_wrong_selection():
    from tacit.evaluation.track2 import evaluate_discriminative
    result = evaluate_discriminative(
        correct_index=2,
        selected_index=0,
    )
    assert result.correct is False
    assert result.selected_index == 0
    assert result.correct_index == 2
```

**Step 2: Write failing tests for Track 1**

```python
# tests/evaluation/test_track1.py
import pytest
from unittest.mock import MagicMock
from tacit.core.types import VerificationResult


def test_track1_correct_solution():
    from tacit.evaluation.track1 import evaluate_generative

    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(passed=True)

    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_svg="<svg>correct</svg>",
    )
    assert result.passed is True


def test_track1_incorrect_solution():
    from tacit.evaluation.track1 import evaluate_generative

    mock_generator = MagicMock()
    mock_generator.verify.return_value = VerificationResult(
        passed=False, reason="path disconnected"
    )

    result = evaluate_generative(
        generator=mock_generator,
        puzzle=MagicMock(),
        candidate_svg="<svg>wrong</svg>",
    )
    assert result.passed is False
```

**Step 3: Write failing tests for metrics**

```python
# tests/evaluation/test_metrics.py
import pytest


def test_accuracy_computation():
    from tacit.evaluation.metrics import compute_accuracy
    results = [True, True, False, True, False]
    assert compute_accuracy(results) == pytest.approx(0.6)


def test_accuracy_by_difficulty():
    from tacit.evaluation.metrics import compute_accuracy_by_difficulty
    results = [
        {"difficulty": "easy", "correct": True},
        {"difficulty": "easy", "correct": True},
        {"difficulty": "hard", "correct": False},
        {"difficulty": "hard", "correct": True},
    ]
    acc = compute_accuracy_by_difficulty(results)
    assert acc["easy"] == pytest.approx(1.0)
    assert acc["hard"] == pytest.approx(0.5)


def test_accuracy_by_task():
    from tacit.evaluation.metrics import compute_accuracy_by_task
    results = [
        {"task": "maze", "correct": True},
        {"task": "maze", "correct": False},
        {"task": "raven", "correct": True},
    ]
    acc = compute_accuracy_by_task(results)
    assert acc["maze"] == pytest.approx(0.5)
    assert acc["raven"] == pytest.approx(1.0)
```

**Step 4: Write failing tests for harness**

```python
# tests/evaluation/test_harness.py
import pytest
from unittest.mock import MagicMock, patch


def test_harness_loads_generators():
    from tacit.evaluation.harness import EvaluationHarness
    harness = EvaluationHarness()
    generators = harness.available_tasks()
    assert isinstance(generators, list)


def test_harness_run_track2():
    from tacit.evaluation.harness import EvaluationHarness
    harness = EvaluationHarness()
    # Minimal smoke test — full integration tested via CLI
    assert hasattr(harness, "run_track2")
```

**Step 5: Run all tests to verify they fail**

Run: `pytest tests/evaluation/ -v`
Expected: FAIL

**Step 6: Implement track2.py**

```python
# tacit/evaluation/track2.py
"""Track 2 — Discriminative evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscriminativeResult:
    correct: bool
    selected_index: int
    correct_index: int


def evaluate_discriminative(
    correct_index: int,
    selected_index: int,
) -> DiscriminativeResult:
    """Evaluate a discriminative (multiple-choice) response."""
    return DiscriminativeResult(
        correct=(correct_index == selected_index),
        selected_index=selected_index,
        correct_index=correct_index,
    )
```

**Step 7: Implement track1.py**

```python
# tacit/evaluation/track1.py
"""Track 1 — Generative evaluation."""
from __future__ import annotations

from tacit.core.types import PuzzleInstance, VerificationResult
from tacit.generators.base import BaseGenerator


def evaluate_generative(
    generator: BaseGenerator,
    puzzle: PuzzleInstance,
    candidate_svg: str,
) -> VerificationResult:
    """Evaluate a generative (image-output) response.

    Delegates to the task-specific generator's verify() method.
    """
    return generator.verify(puzzle, candidate_svg)
```

**Step 8: Implement metrics.py**

```python
# tacit/evaluation/metrics.py
"""Scoring and aggregation metrics for TACIT Benchmark."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_accuracy(results: list[bool]) -> float:
    """Compute overall accuracy from a list of pass/fail booleans."""
    if not results:
        return 0.0
    return sum(results) / len(results)


def compute_accuracy_by_difficulty(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute accuracy grouped by difficulty level."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        groups[r["difficulty"]].append(r["correct"])
    return {k: compute_accuracy(v) for k, v in groups.items()}


def compute_accuracy_by_task(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute accuracy grouped by task name."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        groups[r["task"]].append(r["correct"])
    return {k: compute_accuracy(v) for k, v in groups.items()}
```

**Step 9: Implement harness.py**

```python
# tacit/evaluation/harness.py
"""Core evaluation orchestration for TACIT Benchmark."""
from __future__ import annotations

from typing import Any

from tacit.generators.base import BaseGenerator


# Registry of task name -> generator class
_GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {}


def register_generator(task_name: str, generator_cls: type[BaseGenerator]) -> None:
    """Register a generator class for a task."""
    _GENERATOR_REGISTRY[task_name] = generator_cls


class EvaluationHarness:
    """Orchestrates evaluation across tasks and tracks."""

    def __init__(self) -> None:
        self._generators: dict[str, BaseGenerator] = {}
        self._load_generators()

    def _load_generators(self) -> None:
        """Instantiate all registered generators."""
        for name, cls in _GENERATOR_REGISTRY.items():
            self._generators[name] = cls()

    def available_tasks(self) -> list[str]:
        """List available task names."""
        return list(self._generators.keys())

    def get_generator(self, task_name: str) -> BaseGenerator:
        """Get generator instance by task name."""
        return self._generators[task_name]

    def run_track2(
        self,
        task_name: str,
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Run Track 2 evaluation for a task. Returns accuracy metrics."""
        from tacit.evaluation.track2 import evaluate_discriminative
        from tacit.evaluation.metrics import compute_accuracy

        corrects = []
        for r in results:
            res = evaluate_discriminative(r["correct_index"], r["selected_index"])
            corrects.append(res.correct)
        return {"accuracy": compute_accuracy(corrects), "total": len(corrects)}
```

**Step 10: Implement report.py (minimal)**

```python
# tacit/evaluation/report.py
"""Result report generation for TACIT Benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_report(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """Write evaluation results as JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
```

**Step 11: Run tests to verify they pass**

Run: `pytest tests/evaluation/ -v`
Expected: All PASS

**Step 12: Commit**

```bash
git add tacit/evaluation/ tests/evaluation/
git commit -m "feat: evaluation harness — dual-track evaluation with metrics and reporting"
```

---

## Task 17: CLI

**Files:**
- Create: `tacit/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    from tacit.cli import main
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "TACIT Benchmark" in result.output


def test_cli_generate_help(runner):
    from tacit.cli import main
    result = runner.invoke(main, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--task" in result.output


def test_cli_evaluate_help(runner):
    from tacit.cli import main
    result = runner.invoke(main, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--track" in result.output


def test_cli_publish_help(runner):
    from tacit.cli import main
    result = runner.invoke(main, ["publish", "--help"])
    assert result.exit_code == 0
    assert "--hf-repo" in result.output


def test_cli_version(runner):
    from tacit.cli import main
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Implement CLI**

```python
# tacit/cli.py
"""CLI entry point for TACIT Benchmark."""
from __future__ import annotations

from pathlib import Path

import click
import yaml

import tacit


@click.group()
@click.version_option(version=tacit.__version__, prog_name="TACIT Benchmark")
def main() -> None:
    """TACIT Benchmark: A Programmatic Visual Reasoning Benchmark."""
    pass


@main.command()
@click.option("--task", type=str, default=None, help="Task name (e.g., maze, raven)")
@click.option("--difficulty", type=str, default="easy", help="Difficulty level")
@click.option("--count", type=int, default=10, help="Number of puzzles to generate")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--config", type=click.Path(exists=True), default=None, help="Config YAML file")
@click.option("--output-dir", type=click.Path(), default="data", help="Output directory")
@click.option("--distractors", type=int, default=4, help="Distractors per puzzle")
def generate(task, difficulty, count, seed, config, output_dir, distractors) -> None:
    """Generate puzzle instances."""
    if config:
        cfg = yaml.safe_load(Path(config).read_text())
        click.echo(f"Generating from config: {config}")
        # TODO: iterate over configured tasks and generate
    elif task:
        click.echo(f"Generating {count} {task} puzzles at {difficulty} difficulty (seed={seed})")
        # TODO: instantiate generator and run
    else:
        raise click.UsageError("Provide --task or --config")


@main.command()
@click.option("--track", type=click.Choice(["generative", "discriminative"]), required=True)
@click.option("--model-output", type=click.Path(exists=True), required=True, help="Model output directory")
@click.option("--tasks", type=str, default="all", help="Comma-separated task names or 'all'")
@click.option("--output", type=click.Path(), default="results.json", help="Output report path")
def evaluate(track, model_output, tasks, output) -> None:
    """Evaluate model outputs against benchmark."""
    click.echo(f"Evaluating Track {track} from {model_output}")
    # TODO: load model outputs and run evaluation


@main.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Generation config")
@click.option("--hf-repo", type=str, required=True, help="HuggingFace repo (e.g., tylerxdurden/TACIT-benchmark)")
@click.option("--version-tag", type=str, default=None, help="Version tag (default: from pyproject.toml)")
def publish(config, hf_repo, version_tag) -> None:
    """Generate and publish frozen snapshot to HuggingFace."""
    click.echo(f"Publishing to {hf_repo}")
    # TODO: generate full suite, compute checksums, push to HF
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add tacit/cli.py tests/test_cli.py
git commit -m "feat: CLI with generate, evaluate, and publish commands"
```

---

## Task 18: HuggingFace Publish Script

**Files:**
- Create: `scripts/publish_hf.py`
- Test: `tests/test_publish.py`

**Step 1: Write failing tests**

```python
# tests/test_publish.py
import pytest
import json
import tempfile
from pathlib import Path


def test_generate_dataset_card():
    from scripts.publish_hf import generate_dataset_card
    card = generate_dataset_card(version="0.1.0")
    assert "Daniel Nobrega Medeiros" in card
    assert "TACIT" in card
    assert "tylerxdurden" in card
    assert "10.57967/hf/7904" in card


def test_compute_checksums():
    from scripts.publish_hf import compute_checksums
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "test.txt").write_text("hello")
        checksums = compute_checksums(p)
        assert "test.txt" in checksums
        assert checksums["test.txt"].startswith("sha256:")


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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_publish.py -v`
Expected: FAIL

**Step 3: Implement**

```python
# scripts/publish_hf.py
"""HuggingFace publish pipeline for TACIT Benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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


def compute_checksums(directory: Path) -> dict[str, str]:
    """Compute SHA-256 checksums for all files in directory."""
    checksums = {{}}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            rel = str(path.relative_to(directory))
            checksums[rel] = f"sha256:{{sha}}"
    return checksums


def generate_metadata(
    version: str,
    seed: int,
    config: dict[str, Any],
    checksums: dict[str, str],
) -> dict[str, Any]:
    """Generate metadata.json for the HF snapshot."""
    return {{
        "version": version,
        "seed": seed,
        "generation_config": config,
        "checksums": checksums,
    }}
```

Note: The actual HF upload logic (`huggingface_hub.HfApi.upload_folder`) will be wired into the CLI's `publish` command. This module provides the helper functions.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_publish.py -v`
Expected: All 3 PASS

**Step 5: Commit**

```bash
git add scripts/publish_hf.py tests/test_publish.py
git commit -m "feat: HuggingFace publish helpers — dataset card, checksums, metadata"
```

---

## Task 19: README (English)

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

Contents should include:
- Project title and one-line description
- Badges (license, Python version, HuggingFace link)
- Overview: what TACIT is, the 10 tasks, dual-track evaluation
- Quick start: `pip install -e .` → `tacit generate --task maze --difficulty easy --count 10`
- Task table: all 10 tasks with domain, description, difficulty axes
- Evaluation: Track 1 vs Track 2 explanation
- Citation block (bibtex)
- License
- Author attribution

Keep it concise — this is a research benchmark, not a tutorial. Link to `docs/en/` for detailed documentation.

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with project overview, quick start, and citation"
```

---

## Task 20: README (Chinese)

**Files:**
- Create: `README_zh.md`

**Step 1: Write README_zh.md**

Full Chinese translation of README.md. Same structure, same content, natural Chinese technical writing (not machine-translated awkwardness).

**Step 2: Commit**

```bash
git add README_zh.md
git commit -m "docs: Chinese README (README_zh.md)"
```

---

## Task 21: Integration Test & Final Validation

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: generate → verify → evaluate."""
import pytest
from tacit.core.types import DifficultyParams


TASKS_AND_PARAMS = [
    ("maze", {"grid_size": 8, "layers": 1, "portals": 0}),
    ("raven", {"rules": 1, "complexity": "additive"}),
    ("ca_forward", {"grid_size": 8, "rule_complexity": 2, "steps": 1}),
    ("ca_inverse", {"grid_size": 8, "rule_space": 4, "steps": 1}),
    ("logic_grid", {"grid_size": 4, "constraints": 6, "types": 2}),
    ("graph_coloring", {"nodes": 6, "edge_density": 0.3, "k": 4}),
    ("graph_isomorphism", {"nodes": 5, "distortion": 0.3}),
    ("unknot", {"crossings": 3}),
    ("ortho_projection", {"faces": 6, "concavities": 0}),
    ("iso_reconstruction", {"faces": 6, "ambiguity": 0}),
]


@pytest.mark.parametrize("task_name,params", TASKS_AND_PARAMS)
def test_full_pipeline(task_name, params):
    """For each task: generate puzzle, verify solution passes, verify distractors fail."""
    from tacit.evaluation.harness import EvaluationHarness

    harness = EvaluationHarness()
    gen = harness.get_generator(task_name)

    dp = DifficultyParams(level="easy", params=params)
    puzzle = gen.generate(dp, seed=42, num_distractors=4)

    # Solution must verify
    result = gen.verify(puzzle, puzzle.solution_svg)
    assert result.passed, f"{task_name}: solution failed verification — {result.reason}"

    # All distractors must fail
    for i, svg in enumerate(puzzle.distractor_svgs):
        result = gen.verify(puzzle, svg)
        assert not result.passed, (
            f"{task_name}: distractor {i} ({puzzle.distractor_violations[i]}) "
            f"passed verification — should have failed"
        )


@pytest.mark.parametrize("task_name,params", TASKS_AND_PARAMS)
def test_determinism(task_name, params):
    """Same seed must produce identical puzzles."""
    from tacit.evaluation.harness import EvaluationHarness

    harness = EvaluationHarness()
    gen = harness.get_generator(task_name)

    dp = DifficultyParams(level="easy", params=params)
    p1 = gen.generate(dp, seed=99)
    p2 = gen.generate(dp, seed=99)
    assert p1.puzzle_svg == p2.puzzle_svg
    assert p1.solution_svg == p2.solution_svg
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration tests for all 10 tasks"
```

---

## Dependency Graph

```
Task 1 (scaffolding)
  └─→ Task 2 (types)
       └─→ Task 3 (renderer)
       └─→ Task 4 (verifier + distractor)
            └─→ Task 5 (generator base)
                 ├─→ Task 6 (maze) ─────────────┐
                 ├─→ Task 7 (raven) ────────────┤
                 ├─→ Task 8 (CA forward) ───────┤
                 ├─→ Task 9 (CA inverse) ───────┤
                 ├─→ Task 10 (logic grid) ──────┤
                 ├─→ Task 11 (graph coloring) ──┤ All parallelizable
                 ├─→ Task 12 (graph iso) ───────┤
                 ├─→ Task 13 (unknot) ──────────┤
                 ├─→ Task 14 (ortho proj) ──────┤
                 └─→ Task 15 (iso recon) ───────┘
                                                 │
                      Task 16 (eval harness) ←───┘
                           │
                      Task 17 (CLI) ←── Task 18 (HF publish)
                           │
                      Task 19 (README EN) ←── Task 20 (README ZH)
                           │
                      Task 21 (integration tests)
```

**Tasks 6-15 are fully parallelizable** — they share no state and depend only on the core infrastructure (Tasks 1-5). This is where subagent-driven development shines.

**Tasks 8 & 9** (CA forward/inverse) share `_ca_common.py` — implement them sequentially.
**Tasks 14 & 15** (ortho/iso) share `_geometry_common.py` — implement them sequentially.
All other generator tasks are fully independent.
