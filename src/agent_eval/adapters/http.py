"""An HTTP agent adapter that delegates a trial to a remote service.

Beyond a plain POST, this adapter is hardened for real-world endpoints:

- **Response schema validation** — the response body is parsed into a typed
  :class:`AgentResponse`, so malformed payloads fail with a precise message.
- **Retry/backoff** — transient failures (connection errors, timeouts, 5xx)
  are retried with exponential backoff; non-transient failures (4xx, schema
  errors) fail fast.
- **Request IDs** — every call carries an ``X-Request-ID`` header that is also
  recorded in trial metadata for traceability across systems.
- **Agent version metadata** — a version reported by the agent (response field
  or ``X-Agent-Version`` header) is captured into metadata.
- **Error classification** — failures are tagged (``timeout``, ``connection``,
  ``http_status``, ``decode``, ``schema``) instead of one opaque string.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_eval.adapters.base import AgentRunResult
from agent_eval.environments.base import EvalEnvironment
from agent_eval.schemas import Outcome, Task, Transcript, TranscriptStep


class AgentResponse(BaseModel):
    """The expected response body from a remote agent's ``/run`` endpoint.

    Unknown fields are ignored so agents can return extra data without breaking
    validation; the fields below are the contract the harness relies on.
    """

    model_config = ConfigDict(extra="ignore")

    final_output: str = ""
    transcript: list[TranscriptStep] = Field(default_factory=list)
    outcome: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_version: str | None = None


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry policy for transient HTTP failures."""

    attempts: int = 3
    backoff_base: float = 0.2
    backoff_factor: float = 2.0
    backoff_max: float = 5.0

    def delay_for(self, attempt: int) -> float:
        """Backoff delay (seconds) before the given 1-based retry attempt."""
        raw = self.backoff_base * (self.backoff_factor ** (attempt - 1))
        return min(raw, self.backoff_max)


# Error classes the adapter distinguishes, surfaced in trial metadata.
ERROR_TIMEOUT = "timeout"
ERROR_CONNECTION = "connection"
ERROR_HTTP_STATUS = "http_status"
ERROR_DECODE = "decode"
ERROR_SCHEMA = "schema"


class _AdapterError(Exception):
    """Internal carrier for a classified, possibly-retryable adapter error."""

    def __init__(self, kind: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class HTTPAgentAdapter:
    """POSTs a task to a remote ``/run`` endpoint and parses the response.

    Request body::

        {"task_id": "...", "input": {...}, "metadata": {...},
         "request_id": "..."}

    Expected response (extra fields ignored)::

        {"final_output": "...", "transcript": [...], "outcome": {...},
         "metadata": {...}, "agent_version": "..."}
    """

    name = "http"

    def __init__(
        self,
        url: str,
        timeout: float = 60.0,
        retry: RetryPolicy | None = None,
    ) -> None:
        if not url:
            raise ValueError("HTTPAgentAdapter requires a non-empty --agent-url.")
        self._url = url
        self._timeout = timeout
        self._retry = retry or RetryPolicy()

    async def run(self, task: Task, env: EvalEnvironment) -> AgentRunResult:
        request_id = uuid.uuid4().hex
        payload = {
            "task_id": task.id,
            "input": task.input,
            "metadata": task.metadata,
            "request_id": request_id,
        }
        headers = {"X-Request-ID": request_id}

        try:
            data, version_header = await self._request_with_retries(payload, headers)
        except _AdapterError as exc:
            return AgentRunResult(
                error=f"HTTP agent call failed ({exc.kind}): {exc}",
                metadata={"request_id": request_id, "error_kind": exc.kind},
            )

        return self._parse(data, env, request_id, version_header)

    async def _request_with_retries(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[AgentResponse, str | None]:
        """POST with exponential backoff; raise a classified error on failure."""
        last_exc: _AdapterError | None = None
        for attempt in range(1, self._retry.attempts + 1):
            try:
                return await self._request_once(payload, headers)
            except _AdapterError as exc:
                last_exc = exc
                if not exc.retryable or attempt == self._retry.attempts:
                    raise
                await asyncio.sleep(self._retry.delay_for(attempt))
        assert last_exc is not None  # loop always sets it before raising
        raise last_exc

    async def _request_once(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[AgentResponse, str | None]:
        """One POST attempt, mapping failures to classified ``_AdapterError``."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload, headers=headers)
                response.raise_for_status()
                raw = response.json()
        except httpx.TimeoutException as exc:
            raise _AdapterError(ERROR_TIMEOUT, str(exc), retryable=True) from exc
        except httpx.HTTPStatusError as exc:
            # 5xx is transient; 4xx is a client/contract error and fails fast.
            retryable = exc.response.status_code >= 500
            raise _AdapterError(
                ERROR_HTTP_STATUS,
                f"HTTP {exc.response.status_code}",
                retryable=retryable,
            ) from exc
        except httpx.HTTPError as exc:
            raise _AdapterError(ERROR_CONNECTION, str(exc), retryable=True) from exc
        except ValueError as exc:
            raise _AdapterError(ERROR_DECODE, f"invalid JSON: {exc}", retryable=False) from exc

        try:
            parsed = AgentResponse.model_validate(raw)
        except ValidationError as exc:
            raise _AdapterError(ERROR_SCHEMA, str(exc), retryable=False) from exc
        return parsed, response.headers.get("X-Agent-Version")

    def _parse(
        self,
        data: AgentResponse,
        env: EvalEnvironment,
        request_id: str,
        version_header: str | None,
    ) -> AgentRunResult:
        outcome = _coerce_outcome(data.outcome)
        env.set_state(outcome.state)
        metadata = dict(data.metadata)
        metadata["request_id"] = request_id
        version = data.agent_version or version_header
        if version is not None:
            metadata["agent_version"] = version
        return AgentRunResult(
            final_output=data.final_output,
            transcript=Transcript(steps=list(data.transcript)),
            outcome=outcome,
            metadata=metadata,
        )


def _coerce_outcome(raw: Any) -> Outcome:
    """Accept either ``{"state": {...}}`` or a bare state dict."""
    if isinstance(raw, dict):
        if "state" in raw and isinstance(raw["state"], dict):
            return Outcome(state=dict(raw["state"]))
        return Outcome(state=dict(raw))
    return Outcome()
