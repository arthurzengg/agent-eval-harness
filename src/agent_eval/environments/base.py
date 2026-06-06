"""Environment abstraction for isolating trials.

An ``EvalEnvironment`` gives each trial a clean place to run and a place to read
and write observable state. The runner creates a fresh environment per trial to
guarantee isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agent_eval.schemas import Task


class EvalEnvironment(ABC):
    """Base class for trial environments."""

    @abstractmethod
    async def setup(self, task: Task, trial_id: str) -> None:
        """Prepare the environment for a single trial."""

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up after a trial."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return the current observable environment state."""

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Replace the observable environment state."""

    @property
    @abstractmethod
    def workdir(self) -> Path:
        """The working directory for the current trial."""
