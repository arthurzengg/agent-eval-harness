"""The evaluation harness: run tasks, isolate trials, grade, and aggregate.

For each task the runner executes ``k`` trials. Each trial gets a fresh
environment for isolation, runs the agent adapter, records a transcript and
outcome, grades the trial with all configured graders (concurrently), and scores
it. Task and suite aggregates (including pass@k / pass^k) are then computed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import cast

from agent_eval.adapters.base import AgentAdapter
from agent_eval.environments.base import EvalEnvironment
from agent_eval.environments.local_tempdir import LocalTempDirEnvironment
from agent_eval.graders.base import BaseGrader
from agent_eval.metrics import compute_metrics
from agent_eval.registry import grader_registry
from agent_eval.schemas import (
    EvalSuite,
    GraderConfig,
    GraderResult,
    Outcome,
    SuiteResult,
    Task,
    TaskResult,
    Trial,
    TrialResult,
)
from agent_eval.scoring import score_trial

EnvFactory = Callable[[], EvalEnvironment]


class Runner:
    """Orchestrates running a suite against an agent adapter."""

    def __init__(
        self,
        adapter: AgentAdapter,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self._adapter = adapter
        self._env_factory = env_factory or (lambda: LocalTempDirEnvironment())

    async def run_suite(self, suite: EvalSuite) -> SuiteResult:
        task_results = [await self._run_task(suite, task) for task in suite.tasks]
        max_k = max((suite.task_trials(t) for t in suite.tasks), default=1)
        metrics = compute_metrics(task_results, k=max_k)
        return SuiteResult(
            suite=suite.suite,
            scoring_mode=suite.defaults.scoring.mode,
            task_results=task_results,
            metrics=metrics,
        )

    async def _run_task(self, suite: EvalSuite, task: Task) -> TaskResult:
        trials = suite.task_trials(task)
        scoring = suite.task_scoring(task)
        trial_results: list[TrialResult] = []
        for index in range(trials):
            trial = await self._run_trial(task, index)
            grader_results = await self._grade(task, trial)
            score, passed = score_trial(grader_results, scoring)
            trial_results.append(
                TrialResult(trial=trial, grader_results=grader_results, score=score, passed=passed)
            )

        num = len(trial_results) or 1
        return TaskResult(
            task_id=task.id,
            trials=trial_results,
            pass_rate=sum(1 for t in trial_results if t.passed) / num,
            avg_score=sum(t.score for t in trial_results) / num,
        )

    async def _run_trial(self, task: Task, index: int) -> Trial:
        env = self._env_factory()
        trial_id = str(index)
        start = time.perf_counter()
        try:
            await env.setup(task, trial_id)
            result = await self._adapter.run(task, env)
            latency_ms = (time.perf_counter() - start) * 1000.0
            outcome = _merge_state(result.outcome, env)
            return Trial(
                task_id=task.id,
                index=index,
                final_output=result.final_output,
                transcript=result.transcript,
                outcome=outcome,
                metadata=result.metadata,
                latency_ms=latency_ms,
                error=result.error,
            )
        except Exception as exc:  # noqa: BLE001 - record, never crash the suite
            latency_ms = (time.perf_counter() - start) * 1000.0
            return Trial(task_id=task.id, index=index, latency_ms=latency_ms, error=str(exc))
        finally:
            await env.teardown()

    async def _grade(self, task: Task, trial: Trial) -> list[GraderResult]:
        async def run_one(config: GraderConfig) -> GraderResult:
            grader = cast(BaseGrader, grader_registry.create(config.type, config))
            return await grader.grade(task, trial)

        return list(await asyncio.gather(*(run_one(g) for g in task.graders)))


def _merge_state(outcome: Outcome, env: EvalEnvironment) -> Outcome:
    """Fold the environment's final state into the outcome (outcome wins)."""
    try:
        env_state = env.get_state()
    except Exception:  # noqa: BLE001
        env_state = {}
    merged = {**env_state, **outcome.state}
    return Outcome(state=merged)
