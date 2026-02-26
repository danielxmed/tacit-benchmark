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

    def test_svg_output(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        puzzle = unknot_gen.generate(dp, seed=42)
        assert "svg" in puzzle.puzzle_svg.lower()
        assert "svg" in puzzle.solution_svg.lower()

    def test_different_seeds_different_puzzles(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        p1 = unknot_gen.generate(dp, seed=1)
        p2 = unknot_gen.generate(dp, seed=2)
        assert p1.puzzle_svg != p2.puzzle_svg

    def test_generates_distractors(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        puzzle = unknot_gen.generate(dp, seed=42, num_distractors=4)
        assert len(puzzle.distractor_svgs) == 4
        assert len(puzzle.distractor_violations) == 4

    def test_distractors_fail_verification(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        puzzle = unknot_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = unknot_gen.verify(puzzle, svg)
            assert not result.passed

    def test_difficulty_axes(self, unknot_gen):
        axes = unknot_gen.difficulty_axes()
        names = [a.name for a in axes]
        assert "crossings" in names

    def test_metadata_contains_is_unknot(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        puzzle = unknot_gen.generate(dp, seed=42)
        assert "is_unknot" in puzzle.metadata

    def test_higher_crossings(self, unknot_gen):
        dp = DifficultyParams(level="hard", params={"crossings": 7})
        puzzle = unknot_gen.generate(dp, seed=42)
        result = unknot_gen.verify(puzzle, puzzle.solution_svg)
        assert result.passed


class TestUnknotSolvability:
    """Every generated unknot puzzle must verify correctly."""

    def test_easy_all_verify(self, unknot_gen):
        dp = DifficultyParams(level="easy", params={"crossings": 3})
        for seed in range(10):
            puzzle = unknot_gen.generate(dp, seed=seed)
            result = unknot_gen.verify(puzzle, puzzle.solution_svg)
            assert result.passed, f"Unknot seed={seed} solution failed verification"

    def test_medium_all_verify(self, unknot_gen):
        dp = DifficultyParams(level="medium", params={"crossings": 5})
        for seed in range(10):
            puzzle = unknot_gen.generate(dp, seed=seed)
            result = unknot_gen.verify(puzzle, puzzle.solution_svg)
            assert result.passed, f"Unknot seed={seed} solution failed verification"
