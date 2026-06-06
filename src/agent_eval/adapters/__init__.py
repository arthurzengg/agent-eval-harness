"""Agent adapters and their registry registration."""

from agent_eval.adapters.base import AgentAdapter, AgentRunResult
from agent_eval.adapters.echo import EchoAgentAdapter
from agent_eval.adapters.http import HTTPAgentAdapter
from agent_eval.registry import adapter_registry


@adapter_registry.register("echo")
def _make_echo(**_: object) -> EchoAgentAdapter:
    return EchoAgentAdapter()


@adapter_registry.register("http")
def _make_http(agent_url: str = "", timeout: float = 60.0, **_: object) -> HTTPAgentAdapter:
    return HTTPAgentAdapter(url=agent_url, timeout=timeout)


__all__ = ["AgentAdapter", "AgentRunResult", "EchoAgentAdapter", "HTTPAgentAdapter"]
