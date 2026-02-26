# tests/core/test_verifier.py
import pytest


def test_base_verifier_is_abstract():
    from tacit.core.verifier import BaseVerifier
    with pytest.raises(TypeError):
        BaseVerifier()


def test_base_verifier_subclass():
    from tacit.core.verifier import BaseVerifier
    from tacit.core.types import PuzzleInstance, DifficultyParams, VerificationResult

    class DummyVerifier(BaseVerifier):
        def verify(self, puzzle, candidate_svg):
            return VerificationResult(passed=True)

        def extract_structure(self, svg_string):
            return {"dummy": True}

    v = DummyVerifier()
    dp = DifficultyParams(level="easy", params={})
    puzzle = PuzzleInstance(
        task="dummy", puzzle_id="d_0001", seed=1,
        difficulty=dp, puzzle_svg="<svg/>", solution_svg="<svg/>",
        distractor_svgs=[], distractor_violations=[], metadata={},
    )
    result = v.verify(puzzle, "<svg/>")
    assert result.passed is True
