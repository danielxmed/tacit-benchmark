# tests/generators/test_graph_isomorphism.py
import pytest
from tacit.core.types import DifficultyParams
from tacit.core.renderer import svg_string_to_png


@pytest.fixture
def iso_gen():
    from tacit.generators.graph_isomorphism import GraphIsomorphismGenerator
    return GraphIsomorphismGenerator()


class TestGraphIsomorphismGeneration:
    def test_generates_puzzle(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        assert puzzle.task == "graph_isomorphism"

    def test_deterministic(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        p1 = iso_gen.generate(dp, seed=42)
        p2 = iso_gen.generate(dp, seed=42)
        assert p1.puzzle_svg == p2.puzzle_svg

    def test_solution_verifies(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        result = iso_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed

    def test_balanced_positive_negative(self, iso_gen):
        """Over many seeds, roughly half should be isomorphic, half not."""
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        positive = 0
        for seed in range(20):
            puzzle = iso_gen.generate(dp, seed=seed)
            is_iso = puzzle.metadata.get("is_isomorphic")
            if is_iso:
                positive += 1
        assert 5 <= positive <= 15  # roughly balanced


class TestGraphIsomorphismSvg:
    def test_puzzle_svg_contains_svg_tag(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        assert "<svg" in puzzle.puzzle_svg.lower()
        assert "</svg>" in puzzle.puzzle_svg.lower()

    def test_solution_svg_contains_svg_tag(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        assert "<svg" in puzzle.solution_svg.lower()
        assert "</svg>" in puzzle.solution_svg.lower()

    def test_puzzle_svg_has_two_graphs(self, iso_gen):
        """Puzzle SVG should render two side-by-side graphs with labeled groups."""
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        # Should have Graph A and Graph B labels
        assert "Graph A" in puzzle.puzzle_svg
        assert "Graph B" in puzzle.puzzle_svg


class TestGraphIsomorphismVerification:
    def test_wrong_answer_fails(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        # Distractors should be the opposite answer
        for distractor_svg in puzzle.distractor_svgs:
            result = iso_gen.verify(puzzle, svg_string_to_png(distractor_svg))
            assert not result.passed

    def test_arbitrary_png_fails(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        # A blank white PNG should not parse as either answer
        from tacit.core.renderer import create_canvas, svg_to_string
        blank_canvas = create_canvas(200, 200)
        blank_png = svg_string_to_png(svg_to_string(blank_canvas))
        result = iso_gen.verify(puzzle, blank_png)
        assert not result.passed


class TestGraphIsomorphismDifficulty:
    def test_difficulty_axes_declared(self, iso_gen):
        axes = iso_gen.difficulty_axes()
        names = [a.name for a in axes]
        assert "nodes" in names
        assert "distortion" in names

    def test_more_nodes_harder(self, iso_gen):
        """More nodes should still produce valid puzzles."""
        dp = DifficultyParams(level="hard", params={"nodes": 10, "distortion": 0.8})
        puzzle = iso_gen.generate(dp, seed=42)
        assert puzzle.task == "graph_isomorphism"
        result = iso_gen.verify(puzzle, svg_string_to_png(puzzle.solution_svg))
        assert result.passed


class TestGraphIsomorphismMetadata:
    def test_metadata_has_is_isomorphic(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        assert "is_isomorphic" in puzzle.metadata
        assert isinstance(puzzle.metadata["is_isomorphic"], bool)

    def test_metadata_has_node_count(self, iso_gen):
        dp = DifficultyParams(level="easy", params={"nodes": 5, "distortion": 0.3})
        puzzle = iso_gen.generate(dp, seed=42)
        assert "nodes" in puzzle.metadata
        assert puzzle.metadata["nodes"] == 5
