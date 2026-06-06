"""Environment providers for trial isolation."""

from agent_eval.environments.base import EvalEnvironment
from agent_eval.environments.local_tempdir import LocalTempDirEnvironment

__all__ = ["EvalEnvironment", "LocalTempDirEnvironment"]
