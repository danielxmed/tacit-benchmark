"""Centralized task-name to generator-class mapping.

Single source of truth used by the CLI ``generate`` / ``publish`` commands
and by the publish pipeline (``scripts/publish_hf.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tacit.generators.base import BaseGenerator

# module_path, class_name — lazy-imported to keep startup fast.
GENERATOR_CLASSES: dict[str, tuple[str, str]] = {
    "maze": ("tacit.generators.maze", "MazeGenerator"),
    "raven": ("tacit.generators.raven", "RavenGenerator"),
    "ca_forward": ("tacit.generators.ca_forward", "CAForwardGenerator"),
    "ca_inverse": ("tacit.generators.ca_inverse", "CAInverseGenerator"),
    "logic_grid": ("tacit.generators.logic_grid", "LogicGridGenerator"),
    "graph_coloring": ("tacit.generators.graph_coloring", "GraphColoringGenerator"),
    "graph_isomorphism": (
        "tacit.generators.graph_isomorphism",
        "GraphIsomorphismGenerator",
    ),
    "unknot": ("tacit.generators.unknot", "UnknotGenerator"),
    "ortho_projection": (
        "tacit.generators.ortho_projection",
        "OrthoProjectionGenerator",
    ),
    "iso_reconstruction": (
        "tacit.generators.iso_reconstruction",
        "IsoReconstructionGenerator",
    ),
}


def get_generator(task_name: str) -> BaseGenerator:
    """Dynamically import and instantiate a generator by task name.

    Raises ``KeyError`` if *task_name* is not a recognised TACIT task.
    """
    import importlib

    module_path, class_name = GENERATOR_CLASSES[task_name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
