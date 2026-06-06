"""Tests for entry-point plugin discovery."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

import agent_eval.plugins as plugins
from agent_eval.graders.base import BaseGrader
from agent_eval.registry import adapter_registry, grader_registry


@dataclass
class _FakeEntryPoint:
    """Minimal stand-in for importlib.metadata.EntryPoint."""

    name: str
    loader: object

    def load(self) -> object:
        if isinstance(self.loader, Exception):
            raise self.loader
        return self.loader


class _PluginGrader(BaseGrader):
    type = "plugin_grader"

    async def grade(self, task: object, trial: object) -> object:  # pragma: no cover - unused
        return self.result(score=1.0, passed=True, reason="ok")


def _factory(config: object) -> _PluginGrader:
    return _PluginGrader(config)  # type: ignore[arg-type]


@pytest.fixture
def fake_eps(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, list[object]]]:
    """Patch entry-point lookup and clean up any names we register."""
    groups: dict[str, list[object]] = {}

    def fake_iter(group: str) -> list[object]:
        return groups.get(group, [])

    monkeypatch.setattr(plugins, "_iter_entry_points", fake_iter)
    monkeypatch.setattr(plugins, "_loaded", False)
    before_g = set(grader_registry.names())
    before_a = set(adapter_registry.names())
    yield groups
    for name in set(grader_registry.names()) - before_g:
        grader_registry._factories.pop(name, None)
    for name in set(adapter_registry.names()) - before_a:
        adapter_registry._factories.pop(name, None)


def test_discovers_and_registers_plugin(fake_eps: dict[str, list[object]]) -> None:
    fake_eps["agent_eval.graders"] = [_FakeEntryPoint("plugin_grader", _factory)]
    registered = plugins.load_plugins(force=True)
    assert "agent_eval.graders:plugin_grader" in registered
    assert "plugin_grader" in grader_registry


def test_idempotent(fake_eps: dict[str, list[object]]) -> None:
    fake_eps["agent_eval.graders"] = [_FakeEntryPoint("plugin_grader", _factory)]
    plugins.load_plugins(force=True)
    assert plugins.load_plugins() == []  # second call is a no-op


def test_does_not_clobber_builtins(fake_eps: dict[str, list[object]]) -> None:
    # 'exact_match' is a built-in grader; a plugin with the same name is skipped.
    fake_eps["agent_eval.graders"] = [_FakeEntryPoint("exact_match", _factory)]
    registered = plugins.load_plugins(force=True)
    assert "agent_eval.graders:exact_match" not in registered


def test_broken_plugin_warns_and_is_skipped(fake_eps: dict[str, list[object]]) -> None:
    fake_eps["agent_eval.adapters"] = [
        _FakeEntryPoint("broken", ImportError("boom")),
        _FakeEntryPoint("good_adapter", lambda **_: object()),
    ]
    with pytest.warns(RuntimeWarning, match="broken"):
        registered = plugins.load_plugins(force=True)
    assert "agent_eval.adapters:good_adapter" in registered
    assert "broken" not in adapter_registry
