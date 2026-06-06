"""Explicit plugin registry for graders, adapters, and reporters.

This is the only intentionally mutable global state in the framework. New
graders, adapters, and reporters register themselves by name so that the runner
and CLI can resolve them without hard-coded imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from agent_eval.adapters.base import AgentAdapter
    from agent_eval.graders.base import BaseGrader
    from agent_eval.reporters.base import Reporter

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple name -> factory registry with friendly error messages."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that registers a factory under ``name``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Instantiate the factory registered under ``name``."""
        if name not in self._factories:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {available}.")
        return self._factories[name](*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories


# Singletons. Modules import these and call ``.register(...)`` at import time.
# Parameterized so ``create()`` returns the concrete plugin type (no casts needed).
grader_registry: Registry[BaseGrader] = Registry("grader")
adapter_registry: Registry[AgentAdapter] = Registry("adapter")
reporter_registry: Registry[Reporter] = Registry("reporter")
