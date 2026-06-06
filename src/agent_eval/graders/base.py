"""Grader interface and shared helpers.

A grader scores a single trial against a task's success criteria and returns a
``GraderResult`` with a score in [0, 1], a pass/fail flag, and human-readable
reasons. Graders are intentionally small and composable; the runner combines
their weighted results into a task score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_eval.schemas import GraderConfig, GraderResult, Task, Trial


class BaseGrader(ABC):
    """Base class for all graders."""

    type: str = "base"

    def __init__(self, config: GraderConfig) -> None:
        self.config = config
        self.options: dict[str, Any] = config.options()

    @abstractmethod
    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        """Score ``trial`` for ``task``."""

    def result(
        self,
        *,
        score: float,
        passed: bool,
        reason: str = "",
        details: dict[str, Any] | None = None,
        hard_fail: bool = False,
    ) -> GraderResult:
        """Build a ``GraderResult`` carrying this grader's common fields."""
        return GraderResult(
            grader_type=self.type,
            score=max(0.0, min(1.0, score)),
            passed=passed,
            weight=self.config.weight,
            hard_fail=hard_fail,
            enabled=self.config.enabled,
            reason=reason,
            details=details or {},
        )


def get_dot_path(data: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Resolve a dot-separated ``path`` within nested dicts.

    Returns ``(found, value)``. ``found`` is False if any segment is missing.
    """
    current: Any = data
    for segment in path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return False, None
    return True, current
