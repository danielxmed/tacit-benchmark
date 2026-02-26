# tests/core/test_distractor.py
import pytest


def test_base_distractor_generator_is_abstract():
    from tacit.core.distractor import BaseDistractorGenerator
    with pytest.raises(TypeError):
        BaseDistractorGenerator()


def test_distractor_generator_subclass():
    from tacit.core.distractor import BaseDistractorGenerator

    class DummyDistractorGen(BaseDistractorGenerator):
        def generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return "<svg>distractor</svg>", violation_type

        def available_violations(self):
            return ["type_a", "type_b"]

    gen = DummyDistractorGen()
    assert "type_a" in gen.available_violations()


def test_generate_distractor_set():
    from tacit.core.distractor import BaseDistractorGenerator
    import numpy as np

    class DummyDistractorGen(BaseDistractorGenerator):
        def generate_distractor(self, puzzle_data, solution_data, violation_type, rng):
            return f"<svg>{violation_type}</svg>", violation_type

        def available_violations(self):
            return ["type_a", "type_b", "type_c"]

    gen = DummyDistractorGen()
    rng = np.random.default_rng(42)
    svgs, violations = gen.generate_set(
        puzzle_data={}, solution_data={}, count=4, rng=rng,
    )
    assert len(svgs) == 4
    assert len(violations) == 4
    assert all(v in ["type_a", "type_b", "type_c"] for v in violations)
