"""Aggregate per-task results into suite-level metrics.

Key reliability metrics (agent outputs are non-deterministic, so single trials
are misleading):

- ``pass@k``: a task counts as passed if at least one of its k trials passed.
- ``pass^k``: a task counts as passed only if all k of its trials passed.
"""

from __future__ import annotations

from agent_eval.schemas import MetricsSummary, TaskResult, TrialResult


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pass_at_k(task: TaskResult) -> bool:
    """True if at least one trial passed."""
    return any(t.passed for t in task.trials)


def pass_caret_k(task: TaskResult) -> bool:
    """True only if every trial passed (and there was at least one)."""
    return bool(task.trials) and all(t.passed for t in task.trials)


def compute_metrics(task_results: list[TaskResult], k: int) -> MetricsSummary:
    """Aggregate metrics across all tasks and trials."""
    all_trials = [t for tr in task_results for t in tr.trials]
    total_trials = len(all_trials)

    per_task: dict[str, float] = {tr.task_id: tr.pass_rate for tr in task_results}
    per_grader = _per_grader_pass_rate(all_trials)

    num_tasks = len(task_results) or 1
    return MetricsSummary(
        total_tasks=len(task_results),
        total_trials=total_trials,
        k=k,
        pass_rate=_mean([1.0 if t.passed else 0.0 for t in all_trials]),
        pass_at_k=sum(pass_at_k(tr) for tr in task_results) / num_tasks,
        pass_caret_k=sum(pass_caret_k(tr) for tr in task_results) / num_tasks,
        avg_score=_mean([t.score for t in all_trials]),
        avg_latency_ms=_mean([t.trial.latency_ms for t in all_trials]),
        avg_tool_calls=_mean([len(t.trial.transcript.tool_calls()) for t in all_trials]),
        avg_turns=_mean([float(t.trial.transcript.turn_count()) for t in all_trials]),
        error_rate=_mean([1.0 if t.trial.error else 0.0 for t in all_trials]),
        per_task=per_task,
        per_grader=per_grader,
    )


def _per_grader_pass_rate(trials: list[TrialResult]) -> dict[str, float]:
    totals: dict[str, int] = {}
    passed: dict[str, int] = {}
    for trial_result in trials:
        for gr in trial_result.grader_results:
            totals[gr.grader_type] = totals.get(gr.grader_type, 0) + 1
            passed[gr.grader_type] = passed.get(gr.grader_type, 0) + (1 if gr.passed else 0)
    return {g: passed[g] / totals[g] for g in totals}
