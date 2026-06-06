"""A deterministic mock agent adapter.

The echo agent does not call a model. Instead it constructs a transcript and
outcome that satisfy the task's stated success criteria: it emits the tool calls
that the task's ``tool_calls`` grader marks as required (with their params),
mirrors the task's ``expected_outcome`` into the trial outcome, and folds any
``required_phrases`` into its final output. This makes it a stable fixture for
exercising the harness, graders, metrics, and reporters without external calls.
"""

from __future__ import annotations

from typing import Any

from agent_eval.adapters.base import AgentRunResult
from agent_eval.environments.base import EvalEnvironment
from agent_eval.schemas import (
    Outcome,
    Role,
    Task,
    ToolCall,
    ToolResult,
    Transcript,
    TranscriptStep,
)


class EchoAgentAdapter:
    """Deterministic adapter that fulfils a task's declared criteria."""

    name = "echo"

    async def run(self, task: Task, env: EvalEnvironment) -> AgentRunResult:
        steps: list[TranscriptStep] = []

        user_message = str(task.input.get("user_message", "")) or _first_str(task.input)
        steps.append(TranscriptStep(role=Role.user, content=user_message))

        for spec in _required_tools(task):
            name = str(spec["tool"])
            params = dict(spec.get("params") or {})
            steps.append(
                TranscriptStep(
                    role=Role.assistant,
                    content=f"Calling {name}.",
                    tool_call=ToolCall(name=name, arguments=params),
                )
            )
            steps.append(
                TranscriptStep(
                    role=Role.tool,
                    tool_result=ToolResult(name=name, content={"ok": True}),
                )
            )

        final_output = _build_final_output(task)
        steps.append(TranscriptStep(role=Role.assistant, content=final_output))

        outcome = Outcome(state=dict(task.expected_outcome))
        env.set_state(outcome.state)

        return AgentRunResult(
            final_output=final_output,
            transcript=Transcript(steps=steps),
            outcome=outcome,
            metadata={"adapter": self.name},
        )


def _required_tools(task: Task) -> list[dict[str, Any]]:
    """Collect required tool specs from every ``tool_calls`` grader."""
    required: list[dict[str, Any]] = []
    for grader in task.graders:
        if grader.type != "tool_calls":
            continue
        for item in grader.options().get("required", []) or []:
            if isinstance(item, dict) and "tool" in item:
                required.append(item)
    return required


def _required_phrases(task: Task) -> list[str]:
    phrases: list[str] = []
    for grader in task.graders:
        if grader.type == "transcript":
            phrases.extend(grader.options().get("required_phrases", []) or [])
    return phrases


def _build_final_output(task: Task) -> str:
    summary = str(task.reference.get("summary", "")).strip()
    base = summary or "Request handled."
    phrases = _required_phrases(task)
    if phrases:
        base = base + " " + " ".join(phrases)
    return base


def _first_str(data: dict[str, Any]) -> str:
    for value in data.values():
        if isinstance(value, str):
            return value
    return ""
