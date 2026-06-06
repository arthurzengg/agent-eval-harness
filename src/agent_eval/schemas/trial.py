"""Trial-record models (eval runtime).

These capture what happens when an agent runs one attempt at a task: the ordered
transcript, the tool calls and results, the final observable outcome, and the
top-level ``Trial`` record the runner builds per attempt.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_eval.schemas.suite import JSONValue, Role


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

    def input_tokens(self) -> int:
        """Total input/prompt tokens reported across steps."""
        return sum(s.token_usage.input_tokens for s in self.steps if s.token_usage)

    def output_tokens(self) -> int:
        """Total output/completion tokens reported across steps."""
        return sum(s.token_usage.output_tokens for s in self.steps if s.token_usage)

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
