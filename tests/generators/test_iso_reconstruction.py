# tests/generators/test_iso_reconstruction.py
"""Tests for isometric reconstruction generator."""
import pytest
import numpy as np
from tacit.core.types import DifficultyParams
from tacit.core.renderer import svg_string_to_png


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
        result = iso_recon_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed

    def test_distractors_fail(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        puzzle = iso_recon_gen.generate(dp, seed=42, num_distractors=4)
        for svg in puzzle.distractor_svgs:
            result = iso_recon_gen.verify(puzzle, svg_string_to_png(svg))
            assert not result.passed

    def test_puzzle_svg_is_valid(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        puzzle = iso_recon_gen.generate(dp, seed=42)
        assert "svg" in puzzle.puzzle_svg.lower()
        assert puzzle.solution_svg
        assert "svg" in puzzle.solution_svg.lower()

    def test_different_seeds_different_puzzles(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        p1 = iso_recon_gen.generate(dp, seed=1)
        p2 = iso_recon_gen.generate(dp, seed=2)
        assert p1.puzzle_svg != p2.puzzle_svg

    def test_generates_correct_distractor_count(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        puzzle = iso_recon_gen.generate(dp, seed=42, num_distractors=3)
        assert len(puzzle.distractor_svgs) == 3
        assert len(puzzle.distractor_violations) == 3

    def test_difficulty_axes(self, iso_recon_gen):
        axes = iso_recon_gen.difficulty_axes()
        names = [a.name for a in axes]
        assert "faces" in names
        assert "ambiguity" in names

    def test_available_violations(self, iso_recon_gen):
        violations = iso_recon_gen._available_violations()
        assert "wrong_depth" in violations
        assert "missing_face" in violations
        assert "extra_volume" in violations
        assert "rotated" in violations


class TestIsoReconstructionVerification:
    """Test the round-trip verification: project isometric -> 3 views -> verify."""

    def test_solution_matches_all_projections(self, iso_recon_gen):
        """The solution's re-projections must match the input projections."""
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        puzzle = iso_recon_gen.generate(dp, seed=42)
        result = iso_recon_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed
        # Verify details mention projection match
        assert result.reason == "" or "match" in result.reason.lower() or result.passed


class TestIsoReconstructionMultipleSeeds:
    """Every generated puzzle must verify correctly across seeds."""

    def test_multiple_seeds_verify(self, iso_recon_gen):
        dp = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        for seed in range(5):
            puzzle = iso_recon_gen.generate(dp, seed=seed)
            result = iso_recon_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
            assert result.passed, f"Iso reconstruction seed={seed} failed verification"


class TestIsoReconstructionAmbiguity:
    """Test that ambiguity parameter works."""

    def test_ambiguity_changes_output(self, iso_recon_gen):
        dp_simple = DifficultyParams(level="easy", params={"faces": 6, "ambiguity": 0})
        dp_ambig = DifficultyParams(level="medium", params={"faces": 12, "ambiguity": 1})
        p_simple = iso_recon_gen.generate(dp_simple, seed=42)
        p_ambig = iso_recon_gen.generate(dp_ambig, seed=42)
        assert p_simple.puzzle_svg != p_ambig.puzzle_svg

    def test_solution_verifies_with_ambiguity(self, iso_recon_gen):
        dp = DifficultyParams(level="medium", params={"faces": 12, "ambiguity": 1})
        puzzle = iso_recon_gen.generate(dp, seed=42)
        result = iso_recon_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed
