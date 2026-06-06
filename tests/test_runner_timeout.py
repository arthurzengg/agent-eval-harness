import asyncio

import agent_eval.graders  # noqa: F401 - register graders
from agent_eval.adapters.base import AgentRunResult
from agent_eval.environments.base import EvalEnvironment
from agent_eval.runner import Runner
from agent_eval.schemas import EvalSuite, Task


class SlowAdapter:
    """Adapter that sleeps longer than any reasonable timeout."""

    name = "slow"

    async def run(self, task: Task, env: EvalEnvironment) -> AgentRunResult:
        await asyncio.sleep(5.0)
        return AgentRunResult(final_output="done")


def _suite(timeout: float) -> EvalSuite:
    return EvalSuite.model_validate(
        {
            "suite": {"id": "s", "name": "S"},
            "defaults": {"trials": 2, "timeout_seconds": timeout},
            "tasks": [{"id": "t1", "graders": []}],
        }
    )


async def test_slow_trial_times_out_and_is_recorded() -> None:
    result = await Runner(SlowAdapter()).run_suite(_suite(timeout=0.05))
    trials = result.task_results[0].trials

    assert len(trials) == 2
    for tr in trials:
        assert tr.trial.error is not None
        assert "timed out" in tr.trial.error
    # Timeouts surface in the error rate and never raise out of run_suite.
    assert result.metrics.error_rate == 1.0


async def test_per_task_timeout_override() -> None:
    suite = _suite(timeout=10.0)
    suite.tasks[0].timeout_seconds = 0.05  # task-level override beats the generous default
    result = await Runner(SlowAdapter()).run_suite(suite)
    assert all("timed out" in tr.trial.error for tr in result.task_results[0].trials)
