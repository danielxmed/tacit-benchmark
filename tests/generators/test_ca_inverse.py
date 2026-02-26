# tests/generators/test_ca_inverse.py
"""Tests for cellular automata inverse inference generator."""
import pytest
from tacit.core.types import DifficultyParams
from tacit.core.renderer import svg_string_to_png


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
        result = ca_inv_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed

    def test_distractors_fail(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = ca_inv_gen.verify(puzzle, svg_string_to_png(svg))
            assert not result.passed

    def test_inferred_rule_reproduces_output(self, ca_inv_gen):
        """The gold standard: applying the solution rule to state T must produce state T+k."""
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        # The verification itself tests this property
        result = ca_inv_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed


class TestCAInverseSVGContent:
    """Tests for SVG content structure."""

    def test_puzzle_svg_is_valid_svg(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        assert puzzle.puzzle_svg.startswith("<svg")
        assert "</svg>" in puzzle.puzzle_svg

    def test_solution_svg_is_valid_svg(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        assert puzzle.solution_svg.startswith("<svg")
        assert "</svg>" in puzzle.solution_svg

    def test_puzzle_svg_contains_grids(self, ca_inv_gen):
        """Puzzle SVG should show state T and state T+k grids."""
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        assert "rect" in puzzle.puzzle_svg

    def test_solution_svg_contains_rule_table(self, ca_inv_gen):
        """Solution SVG should show the rule table."""
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42)
        # Rule table uses rect elements for colored cells
        assert "rect" in puzzle.solution_svg

    def test_distractor_svgs_differ_from_solution(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        puzzle = ca_inv_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            assert svg != puzzle.solution_svg


class TestCAInverseDifficultyAxes:
    def test_difficulty_axes_defined(self, ca_inv_gen):
        axes = ca_inv_gen.difficulty_axes()
        assert len(axes) >= 3
        names = [a.name for a in axes]
        assert "grid_size" in names
        assert "rule_space" in names
        assert "steps" in names


class TestCAInverseDifferentSeeds:
    def test_different_seeds_produce_different_puzzles(self, ca_inv_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "rule_space": 4, "steps": 1})
        p1 = ca_inv_gen.generate(dp, seed=42)
        p2 = ca_inv_gen.generate(dp, seed=99)
        assert p1.puzzle_svg != p2.puzzle_svg
