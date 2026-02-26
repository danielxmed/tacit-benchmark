"""Isometric Reconstruction Generator.

Given three orthographic projections (front, top, side), reconstruct the
correct isometric view of the 3D solid.

Puzzle: three orthographic projections in engineering drawing layout
Solution: correct isometric view of the 3D solid
Verification: re-project the candidate 3D solid along all 3 axes and
              compare to input projections
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
    render_three_projections_svg,
    encode_projection,
    decode_projection,
    encode_voxel_grid,
    decode_voxel_grid,
    projections_match,
)


# Metadata keys
_META_GRID = "voxel_grid"
_META_FRONT = "front_projection"
_META_TOP = "top_projection"
_META_SIDE = "side_projection"


class IsoReconstructionGenerator(BaseGenerator):
    """Generator for isometric reconstruction puzzles."""

    def __init__(self) -> None:
        super().__init__(task_name="iso_reconstruction")

    def _generate_puzzle(
        self, difficulty: DifficultyParams, rng: np.random.Generator
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        faces = difficulty.params.get("faces", 6)
        ambiguity = difficulty.params.get("ambiguity", 0)

        # Grid size scales with face count
        grid_size = max(5, min(10, faces // 2 + 3))

        grid = generate_voxel_solid(rng, grid_size=grid_size, faces=faces, concavities=0)

        # For higher ambiguity, try to thin the solid so projections become
        # more ambiguous (multiple 3D solids could match). We do this by
        # removing voxels that don't change any of the three projections.
        if ambiguity > 0:
            grid = self._make_ambiguous(grid, ambiguity, rng)

        # Compute the three projections
        front = project_orthographic(grid, "front")
        top = project_orthographic(grid, "top")
        side = project_orthographic(grid, "side")

        puzzle_data = {
            "grid": grid,
            "grid_size": grid_size,
            "front": front,
            "top": top,
            "side": side,
        }
        solution_data = {
            "grid": grid,
        }
        return puzzle_data, solution_data

    @staticmethod
    def _make_ambiguous(
        grid: NDArray[np.bool_],
        level: int,
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        """Remove voxels that are redundant in all three projections.

        This makes the 3D shape sparser while keeping the same silhouettes,
        so multiple 3D shapes could produce the same projections.
        """
        result = grid.copy()
        original_front = project_orthographic(result, "front")
        original_top = project_orthographic(result, "top")
        original_side = project_orthographic(result, "side")

        filled = list(zip(*np.where(result)))
        rng.shuffle(filled)

        removed = 0
        for x, y, z in filled:
            if removed >= level:
                break
            if not result[x, y, z]:
                continue
            # Temporarily remove
            result[x, y, z] = False
            # Check if all projections still match
            if (projections_match(project_orthographic(result, "front"), original_front)
                    and projections_match(project_orthographic(result, "top"), original_top)
                    and projections_match(project_orthographic(result, "side"), original_side)):
                removed += 1
            else:
                result[x, y, z] = True

        return result

    def _generate_puzzle_svg(self, puzzle_data: Any) -> str:
        front = puzzle_data["front"]
        top = puzzle_data["top"]
        side = puzzle_data["side"]
        return render_three_projections_svg(front, top, side, canvas_size=600, cell_size=15.0)

    def _generate_solution_svg(self, puzzle_data: Any, solution_data: Any) -> str:
        grid = solution_data["grid"]
        return render_isometric_svg(grid, canvas_size=400, cell_size=20.0)

    def _generate_distractor(
        self,
        puzzle_data: Any,
        solution_data: Any,
        violation_type: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        grid = solution_data["grid"].copy()

        if violation_type == "wrong_depth":
            distractor_grid = self._distractor_wrong_depth(grid, puzzle_data, rng)
        elif violation_type == "missing_face":
            distractor_grid = self._distractor_missing_face(grid, rng)
        elif violation_type == "extra_volume":
            distractor_grid = self._distractor_extra_volume(grid, rng)
        elif violation_type == "rotated":
            distractor_grid = self._distractor_rotated(grid, rng)
        else:
            distractor_grid = self._distractor_missing_face(grid, rng)

        svg = render_isometric_svg(distractor_grid, canvas_size=400, cell_size=20.0)
        return svg, violation_type

    def _distractor_wrong_depth(
        self,
        grid: NDArray[np.bool_],
        puzzle_data: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        """Modify depth while keeping 2 of 3 projections correct."""
        result = grid.copy()
        gs = grid.shape[0]
        filled = np.argwhere(result)

        if len(filled) > 1:
            # Remove a voxel and add one somewhere else
            idx = int(rng.integers(0, len(filled)))
            x, y, z = filled[idx]
            result[x, y, z] = False

            # Add a voxel at a different depth position
            attempts = 0
            while attempts < 50:
                attempts += 1
                nx = int(rng.integers(0, gs))
                ny = int(rng.integers(0, gs))
                nz = int(rng.integers(0, gs))
                if not result[nx, ny, nz]:
                    result[nx, ny, nz] = True
                    break

        return result

    def _distractor_missing_face(
        self,
        grid: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        """Remove some voxels to create a missing face."""
        result = grid.copy()
        filled = np.argwhere(result)
        if len(filled) > 1:
            n_remove = max(1, len(filled) // 4)
            indices = rng.choice(len(filled), size=min(n_remove, len(filled) - 1), replace=False)
            for idx in indices:
                x, y, z = filled[idx]
                result[x, y, z] = False
        return result

    def _distractor_extra_volume(
        self,
        grid: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        """Add extra voxels to create phantom volume."""
        result = grid.copy()
        gs = grid.shape[0]
        empty = np.argwhere(~result)
        if len(empty) > 0:
            n_add = max(1, len(empty) // 6)
            indices = rng.choice(len(empty), size=min(n_add, len(empty)), replace=False)
            for idx in indices:
                x, y, z = empty[idx]
                result[x, y, z] = True
        return result

    def _distractor_rotated(
        self,
        grid: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        """Rotate the grid 90 degrees around a random axis."""
        result = grid.copy()
        # Choose rotation axis and number of 90-degree rotations
        rot_axes = [(0, 1), (0, 2), (1, 2)]
        ax = rot_axes[int(rng.integers(0, 3))]
        k = int(rng.integers(1, 4))  # 1, 2, or 3 quarter-turns
        result = np.rot90(result, k=k, axes=ax)
        # If rotation produces identical grid, force a modification
        if np.array_equal(result, grid):
            result = self._distractor_missing_face(grid, rng)
        return result

    def _available_violations(self) -> list[str]:
        return ["wrong_depth", "missing_face", "extra_volume", "rotated"]

    def verify(
        self, puzzle: PuzzleInstance, candidate_svg: str
    ) -> VerificationResult:
        """Verify by comparing candidate SVG to solution SVG.

        The canonical check is SVG string equality. Both are deterministically
        rendered from the same voxel data.
        """
        if candidate_svg == puzzle.solution_svg:
            return VerificationResult(
                passed=True,
                reason="Isometric view matches ground truth.",
            )

        return VerificationResult(
            passed=False,
            reason="Candidate isometric view does not match ground truth.",
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
                name="ambiguity",
                min_val=0,
                max_val=10,
                step=1,
                description="Level of ambiguity (how many valid 3D solids share projections).",
            ),
        ]

    def generate(
        self,
        difficulty: DifficultyParams,
        seed: int,
        num_distractors: int = 4,
    ) -> PuzzleInstance:
        """Override to store voxel grid and projections in metadata."""
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
                _META_FRONT: encode_projection(puzzle_data["front"]),
                _META_TOP: encode_projection(puzzle_data["top"]),
                _META_SIDE: encode_projection(puzzle_data["side"]),
            },
        )
