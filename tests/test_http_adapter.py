"""Tests for the hardened HTTP agent adapter."""

from __future__ import annotations

import httpx
import pytest

from agent_eval.adapters.http import (
    ERROR_HTTP_STATUS,
    ERROR_SCHEMA,
    ERROR_TIMEOUT,
    AgentResponse,
    HTTPAgentAdapter,
    RetryPolicy,
)
from agent_eval.environments.local_tempdir import LocalTempDirEnvironment
from agent_eval.schemas import Task


def _task() -> Task:
    return Task(id="t1", input={"q": "hi"})


async def _run(adapter: HTTPAgentAdapter) -> object:
    env = LocalTempDirEnvironment()
    await env.setup(_task(), "0")
    try:
        return await adapter.run(_task(), env)
    finally:
        await env.teardown()


# The adapter builds its own AsyncClient, so patch it to inject a MockTransport.
@pytest.fixture(autouse=True)
def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        transport = getattr(_patch_client, "transport", None)
        if transport is not None:
            kwargs["transport"] = transport
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)


def _use(transport: httpx.MockTransport) -> None:
    _patch_client.transport = transport  # type: ignore[attr-defined]


async def test_valid_response_parsed_with_request_id_and_version() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request_id"] = request.headers["X-Request-ID"]
        return httpx.Response(
            200,
            json={
                "final_output": "done",
                "transcript": [{"role": "assistant", "content": "hi"}],
                "outcome": {"refund": "processed"},
                "agent_version": "1.2.3",
            },
        )

    _use(httpx.MockTransport(handler))
    result = await _run(HTTPAgentAdapter(url="http://agent.test/run"))

    assert result.error is None
    assert result.final_output == "done"
    assert result.metadata["agent_version"] == "1.2.3"
    assert result.metadata["request_id"] == seen["request_id"]
    assert result.outcome.state == {"refund": "processed"}


async def test_version_from_header_when_absent_in_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"final_output": "x"}, headers={"X-Agent-Version": "9.9"})

    _use(httpx.MockTransport(handler))
    result = await _run(HTTPAgentAdapter(url="http://agent.test/run"))
    assert result.metadata["agent_version"] == "9.9"


async def test_schema_error_fails_fast_and_is_classified() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"transcript": "not-a-list"})

    _use(httpx.MockTransport(handler))
    result = await _run(
        HTTPAgentAdapter(url="http://agent.test/run", retry=RetryPolicy(attempts=3))
    )
    assert result.error is not None
    assert result.metadata["error_kind"] == ERROR_SCHEMA
    assert calls["n"] == 1  # schema errors are non-retryable


async def test_5xx_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"final_output": "ok"})

    _use(httpx.MockTransport(handler))
    result = await _run(
        HTTPAgentAdapter(
            url="http://agent.test/run", retry=RetryPolicy(attempts=3, backoff_base=0.0)
        )
    )
    assert result.error is None
    assert result.final_output == "ok"
    assert calls["n"] == 3


async def test_4xx_fails_fast() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    _use(httpx.MockTransport(handler))
    result = await _run(
        HTTPAgentAdapter(
            url="http://agent.test/run", retry=RetryPolicy(attempts=5, backoff_base=0.0)
        )
    )
    assert result.metadata["error_kind"] == ERROR_HTTP_STATUS
    assert calls["n"] == 1


async def test_timeout_is_retried_until_exhausted() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("slow", request=request)

    _use(httpx.MockTransport(handler))
    result = await _run(
        HTTPAgentAdapter(
            url="http://agent.test/run", retry=RetryPolicy(attempts=3, backoff_base=0.0)
        )
    )
    assert result.metadata["error_kind"] == ERROR_TIMEOUT
    assert calls["n"] == 3


def test_retry_policy_backoff_is_capped() -> None:
    policy = RetryPolicy(attempts=10, backoff_base=1.0, backoff_factor=2.0, backoff_max=5.0)
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(10) == 5.0  # capped


def test_agent_response_ignores_unknown_fields() -> None:
    parsed = AgentResponse.model_validate({"final_output": "x", "extra": 1})
    assert parsed.final_output == "x"
