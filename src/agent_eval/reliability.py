"""pass@k / pass^k reliability curves and per-task flakiness.

A single ``pass@k`` number hides how reliability scales with the number of
attempts. This module computes the whole curve -- ``pass@1..N`` and
``pass^1..N`` -- using the standard unbiased combinatorial estimators, plus
per-task flakiness so inconsistent tasks (those that pass on some trials and
fail on others) can be surfaced and fixed.

For a task with ``n`` trials of which ``c`` passed:

- ``pass@k`` (at least one of k sampled trials passes) is estimated as
  ``1 - C(n-c, k) / C(n, k)`` -- the HumanEval estimator.
- ``pass^k`` (all k sampled trials pass) is estimated as
  ``C(c, k) / C(n, k)``.

Both are exact expectations over the choice of which k of the n trials are
sampled, so they are unbiased and need no resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from agent_eval.schemas import SuiteResult, TaskResult


def pass_at_k_estimate(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` for ``c`` of ``n`` trials passing (k <= n)."""
    if k > n or n == 0:
        return 0.0
    if n - c < k:
        # Too few failures to fill a k-subset, so every subset has a pass.
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_caret_k_estimate(n: int, c: int, k: int) -> float:
    """Unbiased ``pass^k`` for ``c`` of ``n`` trials passing (k <= n)."""
    if k > n or n == 0:
        return 0.0
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


@dataclass(frozen=True)
class ReliabilityPoint:
    """One point on a suite reliability curve."""

    k: int
    pass_at_k: float
    pass_caret_k: float
    n_tasks: int  # tasks contributing at this k (those with >= k trials)


def suite_reliability_curve(task_results: list[TaskResult]) -> list[ReliabilityPoint]:
    """Average per-task ``pass@k`` / ``pass^k`` estimators for k = 1..max(n).

    At each ``k`` only tasks with at least ``k`` trials contribute, so the curve
    stays honest when tasks use different trial counts.
    """
    if not task_results:
        return []
    max_k = max(tr.num_trials for tr in task_results)
    points: list[ReliabilityPoint] = []
    for k in range(1, max_k + 1):
        at_k: list[float] = []
        caret_k: list[float] = []
        for tr in task_results:
            n = tr.num_trials
            if n < k:
                continue
            c = tr.num_passed
            at_k.append(pass_at_k_estimate(n, c, k))
            caret_k.append(pass_caret_k_estimate(n, c, k))
        if not at_k:
            continue
        points.append(
            ReliabilityPoint(
                k=k,
                pass_at_k=sum(at_k) / len(at_k),
                pass_caret_k=sum(caret_k) / len(caret_k),
                n_tasks=len(at_k),
            )
        )
    return points


@dataclass(frozen=True)
class TaskFlakiness:
    """How inconsistently a task passes across its trials."""

    task_id: str
    n_trials: int
    n_passed: int
    pass_rate: float
    flakiness: float  # 0 = fully consistent, 1 = maximally split (50/50)
    is_flaky: bool

    @property
    def label(self) -> str:
        if self.n_trials == 0:
            return "no trials"
        if not self.is_flaky:
            return "consistent pass" if self.n_passed == self.n_trials else "consistent fail"
        return f"flaky ({self.n_passed}/{self.n_trials})"


def task_flakiness(task: TaskResult) -> TaskFlakiness:
    """Compute a flakiness score for one task.

    ``flakiness = 4 * p * (1 - p)`` peaks at 1.0 when the pass rate ``p`` is 0.5
    and is 0 when every trial agrees. A task is *flaky* when it both passes and
    fails at least once.
    """
    n = task.num_trials
    c = task.num_passed
    p = c / n if n else 0.0
    return TaskFlakiness(
        task_id=task.task_id,
        n_trials=n,
        n_passed=c,
        pass_rate=p,
        flakiness=4.0 * p * (1.0 - p),
        is_flaky=0 < c < n,
    )


def flakiness_report(task_results: list[TaskResult]) -> list[TaskFlakiness]:
    """Per-task flakiness, most flaky first (then by task id for stability)."""
    rows = [task_flakiness(tr) for tr in task_results]
    rows.sort(key=lambda f: (-f.flakiness, f.task_id))
    return rows


def flaky_tasks(task_results: list[TaskResult]) -> list[TaskFlakiness]:
    """Only the tasks that pass inconsistently across trials."""
    return [f for f in flakiness_report(task_results) if f.is_flaky]


def reliability_curve_for(result: SuiteResult) -> list[ReliabilityPoint]:
    """Convenience wrapper: reliability curve for a whole ``SuiteResult``."""
    return suite_reliability_curve(result.task_results)
