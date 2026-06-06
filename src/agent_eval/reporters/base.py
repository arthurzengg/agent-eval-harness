"""Reporter interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_eval.schemas import SuiteResult


@runtime_checkable
class Reporter(Protocol):
    """Renders a ``SuiteResult`` to some destination."""

    name: str

    def render(self, result: SuiteResult, output_dir: Path) -> Path | None:
        """Render the result; return the written path if any."""
        ...
