"""A local temp-directory environment that isolates each trial."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from agent_eval.environments.base import EvalEnvironment
from agent_eval.schemas import Task


class LocalTempDirEnvironment(EvalEnvironment):
    """Creates a fresh temp directory per trial and deletes it on teardown.

    Set ``keep_workdirs=True`` to preserve the directories for inspection (the
    CLI exposes this via ``--keep-workdirs``).
    """

    def __init__(self, keep_workdirs: bool = False) -> None:
        self._keep = keep_workdirs
        self._workdir: Path | None = None
        self._state: dict[str, Any] = {}

    async def setup(self, task: Task, trial_id: str) -> None:
        prefix = f"agent-eval-{task.id}-{trial_id}-"
        self._workdir = Path(tempfile.mkdtemp(prefix=prefix))
        self._state = {}

    async def teardown(self) -> None:
        if self._workdir is not None and not self._keep:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def get_state(self) -> dict[str, Any]:
        return dict(self._state)

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            raise RuntimeError("Environment not set up; call setup() first.")
        return self._workdir
