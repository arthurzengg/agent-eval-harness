"""Suite-definition models (eval input).

These describe what a user authors in a suite YAML file: the suite metadata,
shared defaults, tasks, and per-task grader configuration.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# A JSON-compatible value. Pydantic validates the structure leniently.
JSONValue = Any


class Role(StrEnum):
    """The author of a transcript step."""

    user = "user"
    assistant = "assistant"
    tool = "tool"
    system = "system"
    environment = "environment"


class ScoringMode(StrEnum):
    """How a task's grader results are combined into a pass/fail decision."""

    weighted = "weighted"
    binary = "binary"


class Scoring(BaseModel):
    """Scoring policy for a task or suite default."""

    model_config = ConfigDict(extra="forbid")

    mode: ScoringMode = ScoringMode.weighted
    pass_threshold: float = Field(0.8, ge=0.0, le=1.0)


class Defaults(BaseModel):
    """Suite-level defaults applied to tasks that do not override them."""

    model_config = ConfigDict(extra="forbid")

    trials: int = Field(1, ge=1)
    timeout_seconds: float = Field(60.0, gt=0.0)
    scoring: Scoring = Field(default_factory=Scoring)


class SuiteMetadata(BaseModel):
    """Identifying information for an eval suite."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    version: str = "0.1.0"


class GraderConfig(BaseModel):
    """Configuration for a single grader on a task.

    Grader-specific fields (``required``, ``forbidden``, ``expect``, ...) are
    accepted via ``extra="allow"`` and validated by each grader implementation.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    weight: float = Field(1.0, ge=0.0)
    enabled: bool = True

    def options(self) -> dict[str, Any]:
        """Return the grader-specific options (everything but the common keys)."""
        common = {"type", "weight", "enabled"}
        return {k: v for k, v in self.model_dump().items() if k not in common}


class Task(BaseModel):
    """A single eval case: inputs plus success criteria."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str = "conversational_agent"
    input: dict[str, JSONValue] = Field(default_factory=dict)
    reference: dict[str, JSONValue] = Field(default_factory=dict)
    expected_outcome: dict[str, JSONValue] = Field(default_factory=dict)
    graders: list[GraderConfig] = Field(default_factory=list)
    trials: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    scoring: Scoring | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class EvalSuite(BaseModel):
    """A collection of related tasks plus shared defaults."""

    model_config = ConfigDict(extra="forbid")

    suite: SuiteMetadata
    defaults: Defaults = Field(default_factory=Defaults)
    tasks: list[Task]

    def task_trials(self, task: Task) -> int:
        """Resolve the trial count for a task, falling back to defaults."""
        return task.trials if task.trials is not None else self.defaults.trials

    def task_scoring(self, task: Task) -> Scoring:
        """Resolve the scoring policy for a task, falling back to defaults."""
        return task.scoring if task.scoring is not None else self.defaults.scoring

    def task_timeout(self, task: Task) -> float:
        """Resolve the per-trial timeout (seconds) for a task, falling back to defaults."""
        return (
            task.timeout_seconds
            if task.timeout_seconds is not None
            else self.defaults.timeout_seconds
        )
