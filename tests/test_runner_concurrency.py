import asyncio

import agent_eval.graders  # noqa: F401 - register graders
from agent_eval.adapters.base import AgentRunResult
from agent_eval.environments.base import EvalEnvironment
from agent_eval.runner import Runner
from agent_eval.schemas import EvalSuite, Task


class ConcurrencyProbeAdapter:
    """Adapter that records the peak number of simultaneously-running trials."""

    name = "probe"

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def run(self, task: Task, env: EvalEnvironment) -> AgentRunResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)  # hold the slot so overlap is observable
            return AgentRunResult(final_output="ok")
        finally:
            self.active -= 1


def _suite(tasks: int, trials: int) -> EvalSuite:
    return EvalSuite.model_validate(
        {
            "suite": {"id": "s", "name": "S"},
            "defaults": {"trials": trials},
            "tasks": [{"id": f"t{i}", "graders": []} for i in range(tasks)],
        }
    )


async def test_concurrency_one_runs_serially() -> None:
    adapter = ConcurrencyProbeAdapter()
    await Runner(adapter, concurrency=1).run_suite(_suite(tasks=3, trials=2))
    assert adapter.peak == 1  # default serial behavior is preserved


async def test_concurrency_cap_is_honored() -> None:
    adapter = ConcurrencyProbeAdapter()
    # 3 tasks x 3 trials = 9 trials, but the cap must hold the peak at 4.
    await Runner(adapter, concurrency=4).run_suite(_suite(tasks=3, trials=3))
    assert adapter.peak == 4


async def test_results_stay_ordered_and_complete() -> None:
    adapter = ConcurrencyProbeAdapter()
    result = await Runner(adapter, concurrency=8).run_suite(_suite(tasks=4, trials=3))

    assert [tr.task_id for tr in result.task_results] == ["t0", "t1", "t2", "t3"]
    for task_result in result.task_results:
        assert [t.trial.index for t in task_result.trials] == [0, 1, 2]
