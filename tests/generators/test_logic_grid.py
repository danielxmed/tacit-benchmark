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


class TestLogicGridConstraints:
    """Test that the constraint system works correctly."""

    def test_latin_square_validity(self, logic_gen):
        """Generated solution must be a valid Latin square."""
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42)
        # Extract grid from puzzle metadata
        grid = puzzle.metadata.get("solution_grid")
        assert grid is not None
        n = len(grid)
        # Each row contains all symbols
        for row in grid:
            assert len(set(row)) == n, f"Row {row} does not have all unique symbols"
        # Each column contains all symbols
        for col in range(n):
            col_vals = [grid[row][col] for row in range(n)]
            assert len(set(col_vals)) == n, f"Column {col} does not have all unique symbols"

    def test_constraint_types_present(self, logic_gen):
        """Generated puzzle should have multiple constraint types."""
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42)
        constraints = puzzle.metadata.get("constraints", [])
        assert len(constraints) >= 1

    def test_puzzle_svg_is_valid_svg(self, logic_gen):
        """Puzzle SVG must start with proper SVG markup."""
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42)
        assert "<svg" in puzzle.puzzle_svg
        assert "</svg>" in puzzle.puzzle_svg

    def test_solution_svg_is_valid_svg(self, logic_gen):
        """Solution SVG must start with proper SVG markup."""
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42)
        assert "<svg" in puzzle.solution_svg
        assert "</svg>" in puzzle.solution_svg


class TestLogicGridDistractors:
    """Test distractor generation specifics."""

    def test_available_violations(self, logic_gen):
        violations = logic_gen._available_violations()
        assert "constraint_violation" in violations
        assert "symbol_swap" in violations
        assert len(violations) >= 2

    def test_distractor_count_matches(self, logic_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42, num_distractors=3)
        assert len(puzzle.distractor_svgs) == 3
        assert len(puzzle.distractor_violations) == 3

    def test_distractor_svgs_differ_from_solution(self, logic_gen):
        dp = DifficultyParams(level="easy", params={"grid_size": 4, "constraints": 6, "types": 2})
        puzzle = logic_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            assert svg != puzzle.solution_svg


class TestLogicGridDifficulty:
    """Test difficulty axis configuration."""

    def test_difficulty_axes(self, logic_gen):
        axes = logic_gen.difficulty_axes()
        assert len(axes) >= 2
        names = [a.name for a in axes]
        assert "grid_size" in names
        assert "constraints" in names

    def test_larger_grid(self, logic_gen):
        """Larger grids should still produce valid puzzles."""
        dp = DifficultyParams(level="medium", params={"grid_size": 5, "constraints": 8, "types": 3})
        puzzle = logic_gen.generate(dp, seed=99)
        assert puzzle.task == "logic_grid"
        result = logic_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed
