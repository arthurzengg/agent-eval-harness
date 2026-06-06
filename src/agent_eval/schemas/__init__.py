"""Pydantic v2 data models for the eval harness.

The models are split by lifecycle into three submodules, all re-exported here so
``from agent_eval.schemas import X`` keeps working regardless of where ``X`` lives:

- :mod:`agent_eval.schemas.suite` -- suite definition (eval input):
  ``EvalSuite`` / ``SuiteMetadata`` / ``Defaults`` / ``Scoring`` / ``Task`` /
  ``GraderConfig`` plus the ``Role`` and ``ScoringMode`` enums.
- :mod:`agent_eval.schemas.trial` -- trial records (eval runtime):
  ``Trial`` / ``Transcript`` / ``TranscriptStep`` / ``ToolCall`` /
  ``ToolResult`` / ``TokenUsage`` / ``Outcome``.
- :mod:`agent_eval.schemas.results` -- grading and aggregation (eval output):
  ``GraderResult`` / ``TrialResult`` / ``TaskResult`` / ``MetricsSummary`` /
  ``SuiteResult``.
"""

from __future__ import annotations

from agent_eval.schemas.results import (
    GraderResult,
    MetricsSummary,
    SuiteResult,
    TaskResult,
    TrialResult,
)
from agent_eval.schemas.suite import (
    Defaults,
    EvalSuite,
    GraderConfig,
    JSONValue,
    Pricing,
    Role,
    Scoring,
    ScoringMode,
    SuiteMetadata,
    Task,
)
from agent_eval.schemas.trial import (
    Outcome,
    TokenUsage,
    ToolCall,
    ToolResult,
    Transcript,
    TranscriptStep,
    Trial,
)

__all__ = [
    # suite (input)
    "Defaults",
    "EvalSuite",
    "GraderConfig",
    "JSONValue",
    "Pricing",
    "Role",
    "Scoring",
    "ScoringMode",
    "SuiteMetadata",
    "Task",
    # trial (runtime)
    "Outcome",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "Transcript",
    "TranscriptStep",
    "Trial",
    # results (output)
    "GraderResult",
    "MetricsSummary",
    "SuiteResult",
    "TaskResult",
    "TrialResult",
]
