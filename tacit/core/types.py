"""Core types for TACIT Benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DifficultyParams:
    """Parameters defining puzzle difficulty for a specific task."""
    level: str
    params: dict[str, Any]


@dataclass(frozen=True)
class DifficultyRange:
    """Describes one difficulty axis for a task."""
    name: str
    min_val: float
    max_val: float
    step: float | None = None
    description: str = ""


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a candidate solution."""
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PuzzleInstance:
    """A single generated puzzle with its solution and distractors."""
    task: str
    puzzle_id: str
    seed: int
    difficulty: DifficultyParams
    puzzle_svg: str
    solution_svg: str
    distractor_svgs: list[str]
    distractor_violations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    puzzle_png: bytes | None = None
    solution_png: bytes | None = None

    def __post_init__(self) -> None:
        if len(self.distractor_svgs) != len(self.distractor_violations):
            raise ValueError(
                f"distractor_svgs ({len(self.distractor_svgs)}) and "
                f"distractor_violations ({len(self.distractor_violations)}) "
                f"must have the same length"
            )


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol that all task generators must implement."""

    def generate(self, difficulty: DifficultyParams, seed: int) -> PuzzleInstance:
        """Generate a puzzle instance with solution and distractors."""
        ...

    def verify(self, puzzle: PuzzleInstance, candidate_png: bytes) -> VerificationResult:
        """Verify a candidate solution against the puzzle."""
        ...

    def difficulty_axes(self) -> list[DifficultyRange]:
        """Declare the difficulty parameters this task supports."""
        ...
