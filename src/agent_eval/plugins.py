"""Discover external graders, adapters, and reporters via entry points.

Built-in plugins register themselves manually in each package's ``__init__``.
This module makes registration *open*: any installed distribution can add its own
graders/adapters/reporters by declaring entry points, without editing this repo.

A distribution publishes plugins in ``pyproject.toml`` like::

    [project.entry-points."agent_eval.graders"]
    my_grader = "my_pkg.graders:MyGraderFactory"

    [project.entry-points."agent_eval.adapters"]
    my_agent = "my_pkg.adapters:make_my_agent"

The entry-point *name* becomes the plugin name (used in suites / ``--agent``),
and the loaded object is the factory the matching registry will call:

- graders: ``(GraderConfig) -> BaseGrader``
- adapters: ``(**kwargs) -> AgentAdapter``
- reporters: ``() -> Reporter``

Discovery is additive (built-ins win on name clashes), idempotent, and resilient:
a plugin that fails to import emits a warning and is skipped rather than aborting
the run.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points

from agent_eval.registry import Registry, adapter_registry, grader_registry, reporter_registry

# Entry-point group -> the registry plugins in that group are added to.
ENTRY_POINT_GROUPS: dict[str, Registry[object]] = {
    "agent_eval.graders": grader_registry,  # type: ignore[dict-item]
    "agent_eval.adapters": adapter_registry,  # type: ignore[dict-item]
    "agent_eval.reporters": reporter_registry,  # type: ignore[dict-item]
}

_loaded = False


def _iter_entry_points(group: str) -> Iterable[EntryPoint]:
    """Return the entry points declared in ``group`` (wrapper for patching)."""
    return entry_points(group=group)


def load_plugins(*, force: bool = False) -> list[str]:
    """Discover and register entry-point plugins. Returns the names registered.

    Idempotent: after the first successful call it is a no-op unless ``force``.
    Names already present in a registry (e.g. built-ins) are left untouched.
    """
    global _loaded
    if _loaded and not force:
        return []

    registered: list[str] = []
    for group, registry in ENTRY_POINT_GROUPS.items():
        for ep in _iter_entry_points(group):
            if ep.name in registry:
                continue  # built-in or already-loaded name wins
            try:
                factory = ep.load()
            except Exception as exc:  # noqa: BLE001 - one bad plugin must not abort
                warnings.warn(
                    f"Failed to load plugin '{ep.name}' from group '{group}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            registry.register_factory(ep.name, factory)
            registered.append(f"{group}:{ep.name}")

    _loaded = True
    return registered
