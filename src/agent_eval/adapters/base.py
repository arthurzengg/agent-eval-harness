"""Agent adapter interface.

An adapter knows how to drive a particular agent implementation for a task and
return a normalized ``AgentRunResult``. New agents are integrated by adding an
adapter and registering it in the adapter registry.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_eval.environments.base import EvalEnvironment
from agent_eval.schemas import Outcome, Task, Transcript


class AgentRunResult(BaseModel):
    """The normalized result of running an agent on a task."""

    model_config = ConfigDict(extra="forbid")

    final_output: str = ""
    transcript: Transcript = Field(default_factory=Transcript)
    outcome: Outcome = Field(default_factory=Outcome)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class AgentAdapter(Protocol):
    """Protocol every agent adapter implements."""

    name: str

    async def run(self, task: Task, env: EvalEnvironment) -> AgentRunResult:
        """Run the agent for one trial of ``task`` inside ``env``."""
        ...


__all__ = ["AgentAdapter", "AgentRunResult"]
