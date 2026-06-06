"""Grading and aggregation models (eval output).

These capture scoring output: per-grader results, per-trial and per-task
aggregates, the suite-wide metrics summary, and the top-level ``SuiteResult``
written to disk and rendered by the reporters.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_eval.schemas.suite import JSONValue, ScoringMode, SuiteMetadata
from agent_eval.schemas.trial import Trial


class GraderResult(BaseModel):
    """The result of running one grader against one trial."""

    model_config = ConfigDict(extra="forbid")

    grader_type: str
    score: float = Field(0.0, ge=0.0, le=1.0)
    passed: bool = False
    weight: float = 1.0
    hard_fail: bool = False
    enabled: bool = True
    reason: str = ""
    details: dict[str, JSONValue] = Field(default_factory=dict)


class TrialResult(BaseModel):
    """A trial together with its grader results and aggregate score."""

    model_config = ConfigDict(extra="forbid")

    trial: Trial
    grader_results: list[GraderResult] = Field(default_factory=list)
    score: float = 0.0
    passed: bool = False


class TaskResult(BaseModel):
    """All trials for a single task plus per-task aggregates."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    trials: list[TrialResult] = Field(default_factory=list)
    pass_rate: float = 0.0
    avg_score: float = 0.0

    @property
    def num_trials(self) -> int:
        return len(self.trials)

    @property
    def num_passed(self) -> int:
        return sum(1 for t in self.trials if t.passed)


class MetricsSummary(BaseModel):
    """Aggregate metrics across a whole suite run."""

    model_config = ConfigDict(extra="forbid")

    total_tasks: int = 0
    total_trials: int = 0
    pass_rate: float = 0.0
    pass_at_k: float = 0.0
    pass_caret_k: float = 0.0
    k: int = 1
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tool_calls: float = 0.0
    avg_turns: float = 0.0
    error_rate: float = 0.0
    per_task: dict[str, float] = Field(default_factory=dict)
    per_grader: dict[str, float] = Field(default_factory=dict)


class SuiteResult(BaseModel):
    """The full result of running a suite: per-task results plus metrics."""

    model_config = ConfigDict(extra="forbid")

    suite: SuiteMetadata
    scoring_mode: ScoringMode = ScoringMode.weighted
    task_results: list[TaskResult] = Field(default_factory=list)
    metrics: MetricsSummary = Field(default_factory=MetricsSummary)
