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
