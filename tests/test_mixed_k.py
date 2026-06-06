"""Tests for pass@k / pass^k handling under mixed trial counts."""

from __future__ import annotations

import pytest

from agent_eval.metrics import compute_metrics
from agent_eval.schemas import EvalSuite, MetricsSummary, TaskResult, Trial, TrialResult


def _task_result(task_id: str, passes: list[bool]) -> TaskResult:
    trials = [
        TrialResult(trial=Trial(task_id=task_id, index=i), passed=p, score=1.0 if p else 0.0)
        for i, p in enumerate(passes)
    ]
    num = len(trials) or 1
    return TaskResult(
        task_id=task_id,
        trials=trials,
        k=len(trials),
        pass_rate=sum(passes) / num,
        avg_score=sum(1.0 if p else 0.0 for p in passes) / num,
    )


def test_consistent_k_label() -> None:
    results = [_task_result("a", [True, True]), _task_result("b", [True, False])]
    m = compute_metrics(results, k=2)
    assert m.consistent_k is True
    assert m.k_min == m.k_max == 2
    assert m.k_label() == "2"


def test_mixed_k_reports_range_not_single_k() -> None:
    # Task a runs 2 trials, task b runs 4 — a single suite-wide k is misleading.
    results = [_task_result("a", [True, True]), _task_result("b", [True, True, False, True])]
    m = compute_metrics(results, k=4)
    assert m.consistent_k is False
    assert m.k_min == 2
    assert m.k_max == 4
    assert m.k_label() == "2..4"


def test_pass_at_k_is_per_task_under_mixed_k() -> None:
    # a: passes at least once -> pass@k; b: never all-pass -> not pass^k.
    results = [_task_result("a", [False, True]), _task_result("b", [True, False, True, True])]
    m = compute_metrics(results, k=4)
    assert m.pass_at_k == 1.0  # both tasks pass at least one trial
    assert m.pass_caret_k == 0.0  # neither task passes every trial


def test_metrics_summary_defaults_consistent() -> None:
    assert MetricsSummary().k_label() == "1"


def _suite(trials_a: int | None, trials_b: int | None, enforce: bool) -> EvalSuite:
    return EvalSuite.model_validate(
        {
            "suite": {"id": "s", "name": "S"},
            "defaults": {"trials": 3, "enforce_consistent_trials": enforce},
            "tasks": [
                {"id": "a", "trials": trials_a},
                {"id": "b", "trials": trials_b},
            ],
        }
    )


def test_trial_count_errors_when_enforced_and_mixed() -> None:
    suite = _suite(2, 4, enforce=True)
    errors = suite.trial_count_errors()
    assert errors and "different trial counts" in errors[0]


def test_no_errors_when_enforced_but_consistent() -> None:
    suite = _suite(3, 3, enforce=True)
    assert suite.trial_count_errors() == []


def test_no_errors_when_not_enforced() -> None:
    suite = _suite(2, 4, enforce=False)
    assert suite.trial_count_errors() == []


def test_resolved_trial_counts_uses_defaults() -> None:
    suite = _suite(None, 5, enforce=False)
    assert suite.resolved_trial_counts() == {"a": 3, "b": 5}


@pytest.mark.parametrize("enforce", [True, False])
def test_resolved_counts_independent_of_enforcement(enforce: bool) -> None:
    suite = _suite(2, 2, enforce=enforce)
    assert set(suite.resolved_trial_counts().values()) == {2}
