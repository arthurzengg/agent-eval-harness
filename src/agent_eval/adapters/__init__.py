"""Agent adapters and their registry registration."""

from agent_eval.adapters.base import AgentAdapter, AgentRunResult
from agent_eval.adapters.echo import EchoAgentAdapter
from agent_eval.adapters.http import HTTPAgentAdapter, RetryPolicy
from agent_eval.registry import adapter_registry


@adapter_registry.register("echo")
def _make_echo(**_: object) -> EchoAgentAdapter:
    return EchoAgentAdapter()


@adapter_registry.register("http")
def _make_http(
    agent_url: str = "",
    timeout: float = 60.0,
    retry_attempts: int = 3,
    retry_backoff: float = 0.2,
    **_: object,
) -> HTTPAgentAdapter:
    return HTTPAgentAdapter(
        url=agent_url,
        timeout=timeout,
        retry=RetryPolicy(attempts=max(1, retry_attempts), backoff_base=max(0.0, retry_backoff)),
    )


__all__ = ["AgentAdapter", "AgentRunResult", "EchoAgentAdapter", "HTTPAgentAdapter", "RetryPolicy"]
