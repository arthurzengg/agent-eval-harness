"""An HTTP agent adapter that delegates a trial to a remote service."""

from __future__ import annotations

from typing import Any

import httpx

from agent_eval.adapters.base import AgentRunResult
from agent_eval.environments.base import EvalEnvironment
from agent_eval.schemas import Outcome, Task, Transcript, TranscriptStep


class HTTPAgentAdapter:
    """POSTs a task to a remote ``/run`` endpoint and parses the response.

    Request body::

        {"task_id": "...", "input": {...}, "metadata": {...}}

    Expected response::

        {"final_output": "...", "transcript": [...], "outcome": {...},
         "metadata": {...}}
    """

    name = "http"

    def __init__(self, url: str, timeout: float = 60.0) -> None:
        if not url:
            raise ValueError("HTTPAgentAdapter requires a non-empty --agent-url.")
        self._url = url
        self._timeout = timeout

    async def run(self, task: Task, env: EvalEnvironment) -> AgentRunResult:
        payload = {
            "task_id": task.id,
            "input": task.input,
            "metadata": task.metadata,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return AgentRunResult(error=f"HTTP agent call failed: {exc}")

        return self._parse(data, env)

    def _parse(self, data: dict[str, Any], env: EvalEnvironment) -> AgentRunResult:
        steps_raw = data.get("transcript") or []
        steps = [TranscriptStep.model_validate(s) for s in steps_raw]
        outcome_raw = data.get("outcome") or {}
        outcome = _coerce_outcome(outcome_raw)
        env.set_state(outcome.state)
        return AgentRunResult(
            final_output=str(data.get("final_output", "")),
            transcript=Transcript(steps=steps),
            outcome=outcome,
            metadata=dict(data.get("metadata", {}) or {}),
        )


def _coerce_outcome(raw: Any) -> Outcome:
    """Accept either ``{"state": {...}}`` or a bare state dict."""
    if isinstance(raw, dict):
        if "state" in raw and isinstance(raw["state"], dict):
            return Outcome(state=dict(raw["state"]))
        return Outcome(state=dict(raw))
    return Outcome()
