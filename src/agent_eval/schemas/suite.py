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


class Pricing(BaseModel):
    """Token pricing used to estimate run cost, in USD per 1M tokens.

    Defaults to zero, so cost is reported as 0 until a suite configures real
    rates for its target model.
    """

    model_config = ConfigDict(extra="forbid")

    input_per_1m: float = Field(0.0, ge=0.0)
    output_per_1m: float = Field(0.0, ge=0.0)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Estimated USD cost for the given input/output token counts."""
        return (
            input_tokens / 1_000_000.0 * self.input_per_1m
            + output_tokens / 1_000_000.0 * self.output_per_1m
        )


class Defaults(BaseModel):
    """Suite-level defaults applied to tasks that do not override them."""

    model_config = ConfigDict(extra="forbid")

    trials: int = Field(1, ge=1)
    timeout_seconds: float = Field(60.0, gt=0.0)
    scoring: Scoring = Field(default_factory=Scoring)
    pricing: Pricing = Field(default_factory=Pricing)
    # When True, ``validate``/``run`` reject suites whose tasks resolve to
    # different trial counts, keeping pass@k / pass^k comparable across tasks.
    enforce_consistent_trials: bool = False


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

    def resolved_trial_counts(self) -> dict[str, int]:
        """Map each task id to its resolved trial count (its own k)."""
        return {task.id: self.task_trials(task) for task in self.tasks}

    def trial_count_errors(self) -> list[str]:
        """Report inconsistent trial counts when the suite enforces consistency.

        Returns an empty list unless ``defaults.enforce_consistent_trials`` is
        set and tasks resolve to more than one distinct trial count.
        """
        if not self.defaults.enforce_consistent_trials:
            return []
        counts = self.resolved_trial_counts()
        distinct = sorted(set(counts.values()))
        if len(distinct) <= 1:
            return []
        detail = ", ".join(f"{tid}={k}" for tid, k in counts.items())
        return [
            "enforce_consistent_trials is set but tasks resolve to different "
            f"trial counts {distinct}: {detail}."
        ]

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
