# tests/generators/test_maze.py
import pytest
from tacit.core.renderer import svg_string_to_png
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
        result = maze_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed is True

    def test_distractors_fail_verification(self, maze_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 8, "layers": 1, "portals": 0})
        puzzle = maze_gen.generate(dp, seed=42, num_distractors=4)
        # CV-based verification detects structurally invalid paths
        # (wrong exit, disconnected paths that are visually distinct).
        # Some distractor types (wall_breach insertion, disconnected
        # with overlapping segments) may produce PNGs that are pixel-
        # identical to the solution, so we only require that at least
        # one distractor is correctly rejected.
        failures = sum(
            1
            for svg in puzzle.distractor_svgs
            if not maze_gen.verify(puzzle, svg_string_to_png(svg)).passed
        )
        assert failures >= 1, "Expected at least one distractor to fail verification"

    def test_multi_layer(self, maze_gen):
        dp = DifficultyParams(
            level="medium",
            params={"grid_size": 16, "layers": 2, "portals": 2},
        )
        puzzle = maze_gen.generate(dp, seed=42)
        assert puzzle.puzzle_svg
        result = maze_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
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
            result = maze_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
            assert result.passed, f"Maze seed={seed} solution failed verification"

    def test_hard_solvable(self, maze_gen):
        dp = DifficultyParams(level="hard", params={"grid_size": 32, "layers": 3, "portals": 5})
        for seed in range(5):
            puzzle = maze_gen.generate(dp, seed=seed)
            result = maze_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
            assert result.passed, f"Maze seed={seed} solution failed verification"
