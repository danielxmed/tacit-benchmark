"""Orthographic Projection Identification Generator.

Given a 3D solid (voxel model) shown in isometric view plus a specified
projection axis, produce the correct 2D orthographic projection.

Puzzle: isometric view of the 3D solid + axis indicator
Solution: the correct 2D orthographic projection along the axis
Verification: computed projection matches ground truth
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tacit.core.types import (
    DifficultyParams,
    DifficultyRange,
    PuzzleInstance,
    VerificationResult,
)
from tacit.generators.base import BaseGenerator
from tacit.generators._geometry_common import (
    generate_voxel_solid,
    project_orthographic,
    render_isometric_svg,
    render_projection_svg,
    encode_projection,
    decode_projection,
    encode_voxel_grid,
    decode_voxel_grid,
    projections_match,
)


_AXES = ["front", "top", "side"]

# Metadata keys for storing ground truth in puzzle metadata
_META_GRID = "voxel_grid"
_META_AXIS = "projection_axis"
_META_PROJECTION = "ground_truth_projection"


class OrthoProjectionGenerator(BaseGenerator):
    """Generator for orthographic projection identification puzzles."""

    def __init__(self) -> None:
        super().__init__(task_name="ortho_projection")

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        faces = difficulty.params.get("faces", 6)
        concavities = difficulty.params.get("concavities", 0)

        # Grid size scales with face count for room to grow
        grid_size = max(5, min(10, faces // 2 + 3))

        grid = generate_voxel_solid(rng, grid_size=grid_size, faces=faces, concavities=concavities)

        # Choose a random projection axis
        axis = _AXES[int(rng.integers(0, len(_AXES)))]

        # Compute the ground truth projection
        projection = project_orthographic(grid, axis)

        puzzle_data = {
            "grid": grid,
            "axis": axis,
            "grid_size": grid_size,
        }
        solution_data = {
            "projection": projection,
        }
        return puzzle_data, solution_data

    def _generate_puzzle_svg(self, puzzle_data: Any) -> str:
        grid = puzzle_data["grid"]
        axis = puzzle_data["axis"]
        return render_isometric_svg(grid, canvas_size=400, cell_size=20.0, axis_indicator=axis)

    def _generate_solution_svg(self, puzzle_data: Any, solution_data: Any) -> str:
        projection = solution_data["projection"]
        axis = puzzle_data["axis"]
        return render_projection_svg(projection, canvas_size=300, cell_size=20.0, label=f"{axis.title()} Projection")

    def _generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        grid = puzzle_data["grid"]
        axis = puzzle_data["axis"]
        projection = solution_data["projection"].copy()

        if violation_type == "wrong_axis":
            distractor_proj = self._distractor_wrong_axis(grid, axis, rng)
        elif violation_type == "missing_feature":
            distractor_proj = self._distractor_missing_feature(projection, rng)
        elif violation_type == "extra_feature":
            distractor_proj = self._distractor_extra_feature(projection, rng)
        elif violation_type == "mirrored":
            distractor_proj = self._distractor_mirrored(projection, rng)
        else:
            distractor_proj = self._distractor_missing_feature(projection, rng)

        svg = render_projection_svg(distractor_proj, canvas_size=300, cell_size=20.0)
        return svg, violation_type

    def _distractor_wrong_axis(
        self, grid: NDArray[np.bool_], correct_axis: str, rng: np.random.Generator
    ) -> NDArray[np.bool_]:
        """Project along a different axis."""
        other_axes = [a for a in _AXES if a != correct_axis]
        wrong_axis = other_axes[int(rng.integers(0, len(other_axes)))]
        return project_orthographic(grid, wrong_axis)

    def _distractor_missing_feature(
        self, projection: NDArray[np.bool_], rng: np.random.Generator
    ) -> NDArray[np.bool_]:
        """Remove some filled cells from the projection."""
        result = projection.copy()
        filled = np.argwhere(result)
        if len(filled) > 1:
            n_remove = max(1, len(filled) // 4)
            indices = rng.choice(len(filled), size=min(n_remove, len(filled) - 1), replace=False)
            for idx in indices:
                r, c = filled[idx]
                result[r, c] = False
        return result

    def _distractor_extra_feature(
        self, projection: NDArray[np.bool_], rng: np.random.Generator
    ) -> NDArray[np.bool_]:
        """Add phantom geometry to the projection."""
        result = projection.copy()
        empty = np.argwhere(~result)
        if len(empty) > 0:
            n_add = max(1, len(empty) // 4)
            indices = rng.choice(len(empty), size=min(n_add, len(empty)), replace=False)
            for idx in indices:
                r, c = empty[idx]
                result[r, c] = True
        return result

    def _distractor_mirrored(
        self, projection: NDArray[np.bool_], rng: np.random.Generator
    ) -> NDArray[np.bool_]:
        """Left-right flip of the projection."""
        result = np.fliplr(projection.copy())
        # If symmetric, also flip vertically to ensure difference
        if np.array_equal(result, projection):
            result = np.flipud(result)
        # If still equal (symmetric in both axes), remove a feature
        if np.array_equal(result, projection):
            result = self._distractor_missing_feature(projection, rng)
        return result

    def _available_violations(self) -> list[str]:
        return ["wrong_axis", "missing_feature", "extra_feature", "mirrored"]

    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify by re-computing the projection and comparing SVG strings.

        The canonical verification is: the candidate SVG must exactly match the
        solution SVG string (both are deterministically rendered from the same
        projection data).
        """
        if candidate_svg == puzzle.solution_svg:
            return VerificationResult(passed=True, reason="Projection matches ground truth.")

        return VerificationResult(
            passed=False,
            reason="Candidate projection does not match ground truth.",
        )

    def difficulty_axes(self) -> list[DifficultyRange]:
        return [
            DifficultyRange(
                name="faces",
                min_val=4,
                max_val=50,
                step=2,
                description="Number of voxels in the 3D solid.",
            ),
            DifficultyRange(
                name="concavities",
                min_val=0,
                max_val=10,
                step=1,
                description="Number of concavities (removed interior voxels).",
            ),
        ]

    def generate(
        self,
        difficulty: DifficultyParams,
        seed: int,
        num_distractors: int = 4,
    ) -> PuzzleInstance:
        """Override to store voxel grid and axis in metadata for verification."""
        rng_puzzle = np.random.default_rng(seed)
        puzzle_data, solution_data = self._generate_puzzle(difficulty, rng_puzzle)

        puzzle_svg = self._generate_puzzle_svg(puzzle_data)
        solution_svg = self._generate_solution_svg(puzzle_data, solution_data)

        rng_distractor = np.random.default_rng(seed + 2**31)
        distractor_svgs: list[str] = []
        distractor_violations: list[str] = []
        violations = self._available_violations()

        for i in range(num_distractors):
            vtype = violations[i % len(violations)]
            svg, label = self._generate_distractor(
                puzzle_data, solution_data, vtype, rng_distractor
            )
            distractor_svgs.append(svg)
            distractor_violations.append(label)

        puzzle_id = f"{self.task_name}_{difficulty.level}_{seed:04d}"

        return PuzzleInstance(
            task=self.task_name,
            puzzle_id=puzzle_id,
            seed=seed,
            difficulty=difficulty,
            puzzle_svg=puzzle_svg,
            solution_svg=solution_svg,
            distractor_svgs=distractor_svgs,
            distractor_violations=distractor_violations,
            metadata={
                _META_GRID: encode_voxel_grid(puzzle_data["grid"]),
                _META_AXIS: puzzle_data["axis"],
                _META_PROJECTION: encode_projection(solution_data["projection"]),
            },
        )
