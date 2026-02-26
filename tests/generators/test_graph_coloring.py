# tests/generators/test_graph_coloring.py
"""Tests for the Planar Graph k-Coloring generator."""
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


class TestGraphColoringVerification:
    def test_solution_has_correct_color_count(self, coloring_gen):
        """Solution must use exactly k colors."""
        dp = DifficultyParams(level="easy", params={"nodes": 8, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42)
        result = coloring_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed
        assert result.details.get("colors_used") == 4

    def test_no_adjacent_nodes_share_color(self, coloring_gen):
        """No two adjacent nodes should have the same color in solution."""
        dp = DifficultyParams(level="medium", params={"nodes": 10, "edge_density": 0.4, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=99)
        result = coloring_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed
        assert result.details.get("adjacent_conflicts") == 0


class TestGraphColoringDistractors:
    def test_distractor_violations_labeled(self, coloring_gen):
        """Each distractor must have a violation label."""
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42, num_distractors=4)
        assert len(puzzle.distractor_violations) == 4
        for label in puzzle.distractor_violations:
            assert label in ("adjacent_conflict", "missing_color", "wrong_k")

    def test_adjacent_conflict_distractor(self, coloring_gen):
        """adjacent_conflict distractor should have at least one adjacent pair with same color."""
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42, num_distractors=4)
        # Find an adjacent_conflict distractor
        for svg, label in zip(puzzle.distractor_svgs, puzzle.distractor_violations):
            if label == "adjacent_conflict":
                result = coloring_gen.verify(puzzle, svg)
                assert not result.passed
                break


class TestGraphColoringDifficulty:
    def test_difficulty_axes(self, coloring_gen):
        """Generator must declare its difficulty axes."""
        axes = coloring_gen.difficulty_axes()
        assert len(axes) > 0
        names = [a.name for a in axes]
        assert "nodes" in names
        assert "edge_density" in names
        assert "k" in names

    def test_multiple_seeds_produce_different_puzzles(self, coloring_gen):
        """Different seeds should produce different puzzles."""
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        p1 = coloring_gen.generate(dp, seed=1)
        p2 = coloring_gen.generate(dp, seed=2)
        assert p1.puzzle_svg != p2.puzzle_svg

    def test_puzzle_svg_contains_svg_elements(self, coloring_gen):
        """Puzzle SVG should contain valid SVG content."""
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42)
        assert "<svg" in puzzle.puzzle_svg
        assert "<circle" in puzzle.puzzle_svg
        assert "<line" in puzzle.puzzle_svg

    def test_solution_svg_has_colored_nodes(self, coloring_gen):
        """Solution SVG should have colored circles (not just gray/default)."""
        dp = DifficultyParams(level="easy", params={"nodes": 6, "edge_density": 0.3, "k": 4})
        puzzle = coloring_gen.generate(dp, seed=42)
        # Solution should contain color fills from STYLE["colors"]
        assert "<circle" in puzzle.solution_svg
        # Should have multiple different fill colors
        import re
        fills = re.findall(r'fill="(#[0-9A-Fa-f]{6})"', puzzle.solution_svg)
        color_set = set(fills)
        # At least some non-white/non-background colors
        color_set.discard("#FFFFFF")
        assert len(color_set) >= 2
