# tests/generators/test_ca_forward.py
"""Tests for cellular automata forward prediction generator."""
import pytest
from tacit.core.types import DifficultyParams
from tacit.core.renderer import svg_string_to_png


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
        result = ca_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed

    def test_distractors_fail(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = ca_gen.verify(puzzle, svg_string_to_png(svg))
            assert not result.passed

    def test_multi_step(self, ca_gen):
        dp = DifficultyParams(level="hard", params={"grid_size": 16, "rule_complexity": 6, "steps": 5})
        puzzle = ca_gen.generate(dp, seed=42)
        result = ca_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed


class TestCAForwardSVGContent:
    """Tests for SVG content structure."""

    def test_puzzle_svg_is_valid_svg(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42)
        assert puzzle.puzzle_svg.startswith("<svg")
        assert "</svg>" in puzzle.puzzle_svg

    def test_solution_svg_is_valid_svg(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42)
        assert puzzle.solution_svg.startswith("<svg")
        assert "</svg>" in puzzle.solution_svg

    def test_puzzle_svg_contains_grid_data(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42)
        # Puzzle SVG should contain grid cells (rendered as rect elements)
        assert "rect" in puzzle.puzzle_svg

    def test_distractor_svgs_differ_from_solution(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        puzzle = ca_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            assert svg != puzzle.solution_svg


class TestCAForwardDifficultyAxes:
    def test_difficulty_axes_defined(self, ca_gen):
        axes = ca_gen.difficulty_axes()
        assert len(axes) >= 3
        names = [a.name for a in axes]
        assert "grid_size" in names
        assert "rule_complexity" in names
        assert "steps" in names


class TestCAForwardDifferentSeeds:
    def test_different_seeds_produce_different_puzzles(self, ca_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_complexity": 2, "steps": 1})
        p1 = ca_gen.generate(dp, seed=42)
        p2 = ca_gen.generate(dp, seed=99)
        assert p1.puzzle_svg != p2.puzzle_svg
