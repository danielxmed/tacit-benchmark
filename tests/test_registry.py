"""Tests for tacit.generators.registry."""
from __future__ import annotations

import pytest

from tacit.generators.registry import GENERATOR_CLASSES, get_generator


ALL_TASKS = [
    "maze",
    "raven",
    "ca_forward",
    "ca_inverse",
    "logic_grid",
    "graph_coloring",
    "graph_isomorphism",
    "unknot",
    "ortho_projection",
    "iso_reconstruction",
]


class TestGeneratorRegistry:
    """Registry must map all 10 tasks to their generator classes."""

    def test_all_ten_tasks_registered(self):
        assert sorted(GENERATOR_CLASSES.keys()) == sorted(ALL_TASKS)

    def test_registry_has_exactly_ten_entries(self):
        assert len(GENERATOR_CLASSES) == 10

    def test_unknown_task_raises_key_error(self):
        with pytest.raises(KeyError):
            get_generator("nonexistent_task")

    @pytest.mark.parametrize("task_name", ALL_TASKS)
    def test_get_generator_returns_instance(self, task_name):
        from tacit.generators.base import BaseGenerator

        gen = get_generator(task_name)
        assert isinstance(gen, BaseGenerator)
        assert gen.task_name == task_name
