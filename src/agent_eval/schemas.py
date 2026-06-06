"""Pydantic v2 data models for the eval harness.

These models map directly onto the core eval concepts:

- ``EvalSuite`` / ``SuiteMetadata`` -- a collection of related tasks.
- ``Task`` -- a single eval case with inputs and success criteria.
- ``Trial`` -- one attempt at a task (multiple per task, since agents are
  non-deterministic).
- ``Transcript`` / ``TranscriptStep`` / ``ToolCall`` / ``ToolResult`` -- the
  complete record of a trial.
- ``Outcome`` -- the final observable state after a trial.
- ``GraderConfig`` / ``GraderResult`` -- scoring configuration and results.
- ``TaskResult`` / ``SuiteResult`` / ``MetricsSummary`` -- aggregated results.
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


# --------------------------------------------------------------------------- #
# Suite definition (input)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Trial records (runtime)
# --------------------------------------------------------------------------- #


class ToolCall(BaseModel):
    """A single tool/function call made by the agent."""

    model_config = ConfigDict(extra="allow")

    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    call_id: str | None = None


class ToolResult(BaseModel):
    """The result returned for a tool call."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    call_id: str | None = None
    content: JSONValue = None
    error: str | None = None


class TokenUsage(BaseModel):
    """Token accounting for a step, when the agent reports it."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TranscriptStep(BaseModel):
    """A single step in a trial transcript."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: JSONValue = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    timestamp: float | None = None
    duration_ms: float | None = None
    token_usage: TokenUsage | None = None
    error: str | None = None


class Transcript(BaseModel):
    """The complete ordered record of a trial."""

    model_config = ConfigDict(extra="forbid")

    steps: list[TranscriptStep] = Field(default_factory=list)

    def tool_calls(self) -> list[ToolCall]:
        """All tool calls in order of appearance."""
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    def turn_count(self) -> int:
        """Number of assistant/user conversational turns."""
        return sum(1 for s in self.steps if s.role in (Role.assistant, Role.user))

    def error_count(self) -> int:
        """Number of steps that recorded an error."""
        return sum(1 for s in self.steps if s.error)

    def total_duration_ms(self) -> float:
        """Sum of per-step durations that were recorded."""
        return sum(s.duration_ms for s in self.steps if s.duration_ms is not None)

    def total_tokens(self) -> int:
        return sum(s.token_usage.total_tokens for s in self.steps if s.token_usage)


class Outcome(BaseModel):
    """The final observable state after a trial."""

    model_config = ConfigDict(extra="allow")

    state: dict[str, JSONValue] = Field(default_factory=dict)


class Trial(BaseModel):
    """One attempt at a task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    index: int
    final_output: str = ""
    transcript: Transcript = Field(default_factory=Transcript)
    outcome: Outcome = Field(default_factory=Outcome)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- #
# Grading and aggregation (output)
# --------------------------------------------------------------------------- #


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
