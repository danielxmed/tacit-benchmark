# tests/generators/test_ortho_projection.py
"""Tests for orthographic projection identification generator."""
import pytest
import numpy as np
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

    def test_puzzle_svg_is_valid(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        puzzle = ortho_gen.generate(dp, seed=42)
        assert "svg" in puzzle.puzzle_svg.lower()
        assert puzzle.solution_svg
        assert "svg" in puzzle.solution_svg.lower()

    def test_different_seeds_different_puzzles(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        p1 = ortho_gen.generate(dp, seed=1)
        p2 = ortho_gen.generate(dp, seed=2)
        assert p1.puzzle_svg != p2.puzzle_svg

    def test_generates_correct_distractor_count(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        puzzle = ortho_gen.generate(dp, seed=42, num_distractors=3)
        assert len(puzzle.distractor_svgs) == 3
        assert len(puzzle.distractor_violations) == 3

    def test_difficulty_axes(self, ortho_gen):
        axes = ortho_gen.difficulty_axes()
        names = [a.name for a in axes]
        assert "faces" in names
        assert "concavities" in names

    def test_available_violations(self, ortho_gen):
        violations = ortho_gen._available_violations()
        assert "wrong_axis" in violations
        assert "missing_feature" in violations
        assert "extra_feature" in violations
        assert "mirrored" in violations


class TestOrthoProjectionWithConcavities:
    """Test that concavities parameter affects the generated solid."""

    def test_concavity_increases_complexity(self, ortho_gen):
        dp_simple = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        dp_complex = DifficultyParams(level="medium", params={"faces": 12, "concavities": 2})
        p_simple = ortho_gen.generate(dp_simple, seed=42)
        p_complex = ortho_gen.generate(dp_complex, seed=42)
        # More complex should produce different SVGs
        assert p_simple.puzzle_svg != p_complex.puzzle_svg

    def test_solution_verifies_with_concavities(self, ortho_gen):
        dp = DifficultyParams(level="medium", params={"faces": 12, "concavities": 2})
        puzzle = ortho_gen.generate(dp, seed=42)
        result = ortho_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed


class TestOrthoProjectionMultipleSeeds:
    """Every generated puzzle must verify correctly across seeds."""

    def test_multiple_seeds_verify(self, ortho_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "concavities": 0})
        for seed in range(5):
            puzzle = ortho_gen.generate(dp, seed=seed)
            result = ortho_gen.verify(puzzle, puzzle.solution_svg)
            assert result.passed, f"Ortho projection seed={seed} failed verification"
