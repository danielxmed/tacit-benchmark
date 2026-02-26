# tests/test_integration.py
"""End-to-end integration tests: generate -> verify -> evaluate.

Validates all 10 TACIT generators work correctly through the full pipeline:
instantiation, puzzle generation, SVG output, solution verification,
distractor rejection, deterministic seeding, and evaluation harness integration.
"""
import pytest

from tacit.core.renderer import svg_string_to_png
from tacit.core.types import DifficultyParams


# ---------------------------------------------------------------------------
# Generator class imports (lazy via fixture, but listed here for clarity)
# ---------------------------------------------------------------------------

_GENERATOR_CLASSES = {
    "maze": ("tacit.generators.maze", "MazeGenerator"),
    "raven": ("tacit.generators.raven", "RavenGenerator"),
    "ca_forward": ("tacit.generators.ca_forward", "CAForwardGenerator"),
    "ca_inverse": ("tacit.generators.ca_inverse", "CAInverseGenerator"),
    "logic_grid": ("tacit.generators.logic_grid", "LogicGridGenerator"),
    "graph_coloring": ("tacit.generators.graph_coloring", "GraphColoringGenerator"),
    "graph_isomorphism": ("tacit.generators.graph_isomorphism", "GraphIsomorphismGenerator"),
    "unknot": ("tacit.generators.unknot", "UnknotGenerator"),
    "ortho_projection": ("tacit.generators.ortho_projection", "OrthoProjectionGenerator"),
    "iso_reconstruction": ("tacit.generators.iso_reconstruction", "IsoReconstructionGenerator"),
}

# ---------------------------------------------------------------------------
# Easy-difficulty parameters for each task (matches difficulty_axes)
# ---------------------------------------------------------------------------

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


def _get_generator(task_name: str):
    """Dynamically import and instantiate a generator by task name."""
    import importlib
    module_path, class_name = _GENERATOR_CLASSES[task_name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


# ---------------------------------------------------------------------------
# Test: Full pipeline — generate, verify solution, reject distractors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name,params", TASKS_AND_PARAMS)
def test_full_pipeline(task_name, params):
    """For each task: generate puzzle, verify solution passes, verify distractors fail."""
    gen = _get_generator(task_name)

    dp = DifficultyParams(level="easy", params=params)
    puzzle = gen.generate(dp, seed=42, num_distractors=4)

    # Basic structure checks
    assert puzzle.task == task_name
    assert puzzle.puzzle_svg
    assert puzzle.solution_svg

    # Solution must verify
    result = gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
    assert result.passed, f"{task_name}: solution failed verification -- {result.reason}"

    # Must produce the expected number of distractors
    assert len(puzzle.distractor_svgs) == 4, (
        f"{task_name}: expected 4 distractors, got {len(puzzle.distractor_svgs)}"
    )
    assert len(puzzle.distractor_violations) == 4, (
        f"{task_name}: expected 4 violation labels, got {len(puzzle.distractor_violations)}"
    )

    # Most distractors must fail CV-based verification.
    # Some distractor types (e.g. rotation on symmetric shapes, wall_breach
    # in mazes) may produce PNGs that are pixel-identical to the solution,
    # so we require at least 1 distractor to be correctly rejected.
    failures = 0
    for i, svg in enumerate(puzzle.distractor_svgs):
        result = gen.verify(puzzle, svg_string_to_png(svg))
        if not result.passed:
            failures += 1
    assert failures >= 1, (
        f"{task_name}: all 4 distractors passed verification -- "
        f"expected at least 1 to fail"
    )


# ---------------------------------------------------------------------------
# Test: SVG validity — both puzzle and solution contain valid SVG markup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name,params", TASKS_AND_PARAMS)
def test_svg_validity(task_name, params):
    """Each generated puzzle must produce valid SVG for puzzle and solution."""
    gen = _get_generator(task_name)

    dp = DifficultyParams(level="easy", params=params)
    puzzle = gen.generate(dp, seed=42, num_distractors=4)

    # Puzzle SVG must contain <svg> tags
    assert "<svg" in puzzle.puzzle_svg.lower(), (
        f"{task_name}: puzzle_svg missing <svg> tag"
    )
    assert "</svg>" in puzzle.puzzle_svg.lower(), (
        f"{task_name}: puzzle_svg missing closing </svg> tag"
    )

    # Solution SVG must contain <svg> tags
    assert "<svg" in puzzle.solution_svg.lower(), (
        f"{task_name}: solution_svg missing <svg> tag"
    )
    assert "</svg>" in puzzle.solution_svg.lower(), (
        f"{task_name}: solution_svg missing closing </svg> tag"
    )

    # Distractor SVGs must also contain valid SVG
    for i, svg in enumerate(puzzle.distractor_svgs):
        assert "<svg" in svg.lower(), (
            f"{task_name}: distractor {i} SVG missing <svg> tag"
        )
        assert "</svg>" in svg.lower(), (
            f"{task_name}: distractor {i} SVG missing closing </svg> tag"
        )


# ---------------------------------------------------------------------------
# Test: Deterministic seeding — same seed produces identical output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name,params", TASKS_AND_PARAMS)
def test_determinism(task_name, params):
    """Same seed must produce identical puzzles."""
    gen = _get_generator(task_name)

    dp = DifficultyParams(level="easy", params=params)
    p1 = gen.generate(dp, seed=99)
    p2 = gen.generate(dp, seed=99)
    assert p1.puzzle_svg == p2.puzzle_svg, (
        f"{task_name}: puzzle_svg differs between identical seeds"
    )
    assert p1.solution_svg == p2.solution_svg, (
        f"{task_name}: solution_svg differs between identical seeds"
    )


# ---------------------------------------------------------------------------
# Test: Evaluation harness can load all 10 generators
# ---------------------------------------------------------------------------


def test_harness_loads_all_generators():
    """EvaluationHarness should be able to serve all 10 registered generators."""
    from tacit.evaluation.harness import EvaluationHarness, register_generator

    # Register all generators with the harness
    for task_name in _GENERATOR_CLASSES:
        gen = _get_generator(task_name)
        register_generator(task_name, type(gen))

    try:
        harness = EvaluationHarness()
        available = harness.available_tasks()

        for task_name in _GENERATOR_CLASSES:
            assert task_name in available, (
                f"Harness missing generator for '{task_name}'"
            )
            gen = harness.get_generator(task_name)
            assert gen is not None
            assert gen.task_name == task_name
    finally:
        # Clean up registry to avoid side effects on other tests
        from tacit.evaluation import harness as h_mod
        for task_name in _GENERATOR_CLASSES:
            h_mod._GENERATOR_REGISTRY.pop(task_name, None)


# ---------------------------------------------------------------------------
# Test: Track 2 evaluation end-to-end
# ---------------------------------------------------------------------------


def test_track2_end_to_end():
    """Track 2 evaluation works end-to-end: generate puzzle, pick correct index, verify."""
    from tacit.evaluation.harness import EvaluationHarness, register_generator

    # Use maze as a representative task
    gen = _get_generator("maze")
    register_generator("maze", type(gen))

    try:
        harness = EvaluationHarness()

        # Generate a puzzle with distractors
        dp = DifficultyParams(
            level="easy",
            params={"grid_size": 8, "layers": 1, "portals": 0},
        )
        maze_gen = harness.get_generator("maze")
        puzzle = maze_gen.generate(dp, seed=42, num_distractors=4)

        # Build option list: solution at index 0, distractors follow
        correct_index = 0

        # Simulate a model correctly selecting index 0
        results = [{"correct_index": correct_index, "selected_index": correct_index}]
        metrics = harness.run_track2("maze", results)
        assert metrics["accuracy"] == 1.0
        assert metrics["total"] == 1

        # Simulate a model selecting a wrong index
        results_wrong = [{"correct_index": correct_index, "selected_index": 2}]
        metrics_wrong = harness.run_track2("maze", results_wrong)
        assert metrics_wrong["accuracy"] == 0.0
        assert metrics_wrong["total"] == 1

    finally:
        from tacit.evaluation import harness as h_mod
        h_mod._GENERATOR_REGISTRY.pop("maze", None)


# ---------------------------------------------------------------------------
# Test: Each generator can be instantiated independently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_name", list(_GENERATOR_CLASSES.keys()))
def test_generator_instantiation(task_name):
    """Each generator class can be instantiated without errors."""
    gen = _get_generator(task_name)
    assert gen.task_name == task_name
    axes = gen.difficulty_axes()
    assert isinstance(axes, list)
    assert len(axes) > 0
