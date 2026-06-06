"""Tests for pass@k / pass^k reliability curves and flakiness."""

from __future__ import annotations

from rich.console import Console

from agent_eval.reliability import (
    flakiness_report,
    flaky_tasks,
    pass_at_k_estimate,
    pass_caret_k_estimate,
    suite_reliability_curve,
    task_flakiness,
)
from agent_eval.reporters.console_reporter import ConsoleReporter
from agent_eval.reporters.html_reporter import render_html
from agent_eval.schemas import (
    GraderResult,
    MetricsSummary,
    SuiteMetadata,
    SuiteResult,
    TaskResult,
    Trial,
    TrialResult,
)


def _task(task_id: str, passes: list[bool]) -> TaskResult:
    trials = [
        TrialResult(
            trial=Trial(task_id=task_id, index=i),
            grader_results=[GraderResult(grader_type="exact_match", passed=p)],
            passed=p,
        )
        for i, p in enumerate(passes)
    ]
    n = len(passes)
    return TaskResult(
        task_id=task_id,
        trials=trials,
        k=n,
        pass_rate=(sum(passes) / n) if n else 0.0,
    )


def test_pass_at_k_all_pass_is_one() -> None:
    assert pass_at_k_estimate(4, 4, 2) == 1.0


def test_pass_at_k_none_pass_is_zero() -> None:
    assert pass_at_k_estimate(4, 0, 1) == 0.0
    assert pass_at_k_estimate(4, 0, 3) == 0.0


def test_pass_at_k_single_pass_estimator() -> None:
    # 1 of 4 passed: pass@1 = 1/4, and pass@4 = 1 (the subset is all trials).
    assert abs(pass_at_k_estimate(4, 1, 1) - 0.25) < 1e-9
    assert pass_at_k_estimate(4, 1, 4) == 1.0


def test_pass_caret_k_estimator() -> None:
    # 2 of 4 passed: pass^1 = 1/2; pass^2 = C(2,2)/C(4,2) = 1/6; pass^3 = 0.
    assert abs(pass_caret_k_estimate(4, 2, 1) - 0.5) < 1e-9
    assert abs(pass_caret_k_estimate(4, 2, 2) - (1 / 6)) < 1e-9
    assert pass_caret_k_estimate(4, 2, 3) == 0.0


def test_k_greater_than_n_is_zero() -> None:
    assert pass_at_k_estimate(2, 1, 3) == 0.0
    assert pass_caret_k_estimate(2, 1, 3) == 0.0


def test_suite_curve_is_monotonic_in_expected_directions() -> None:
    tasks = [_task("t1", [True, False, True, False]), _task("t2", [True, True, False, True])]
    curve = suite_reliability_curve(tasks)
    assert [p.k for p in curve] == [1, 2, 3, 4]
    # pass@k never decreases with k; pass^k never increases with k.
    for a, b in zip(curve, curve[1:], strict=False):
        assert b.pass_at_k >= a.pass_at_k - 1e-9
        assert b.pass_caret_k <= a.pass_caret_k + 1e-9


def test_suite_curve_respects_mixed_trial_counts() -> None:
    tasks = [_task("t1", [True, False]), _task("t2", [True, True, False, True])]
    curve = suite_reliability_curve(tasks)
    by_k = {p.k: p for p in curve}
    assert by_k[1].n_tasks == 2  # both tasks have >= 1 trial
    assert by_k[3].n_tasks == 1  # only t2 has >= 3 trials


def test_suite_curve_empty() -> None:
    assert suite_reliability_curve([]) == []


def test_task_flakiness_consistent_pass() -> None:
    f = task_flakiness(_task("t", [True, True, True]))
    assert not f.is_flaky
    assert f.flakiness == 0.0
    assert f.label == "consistent pass"


def test_task_flakiness_split_is_maximal() -> None:
    f = task_flakiness(_task("t", [True, False]))
    assert f.is_flaky
    assert abs(f.flakiness - 1.0) < 1e-9
    assert "flaky" in f.label


def test_flaky_tasks_filters_and_sorts() -> None:
    tasks = [
        _task("steady", [True, True, True, True]),
        _task("split", [True, False, True, False]),
        _task("rare", [True, True, True, False]),
    ]
    report = flakiness_report(tasks)
    assert [f.task_id for f in report][0] == "split"  # most flaky first
    flaky = flaky_tasks(tasks)
    assert {f.task_id for f in flaky} == {"split", "rare"}


def _suite_result() -> SuiteResult:
    tasks = [_task("split", [True, False, True, False]), _task("steady", [True, True, True, True])]
    return SuiteResult(
        suite=SuiteMetadata(id="s", name="S"),
        task_results=tasks,
        metrics=MetricsSummary(total_tasks=2, total_trials=8, k_min=4, k_max=4),
    )


def test_console_reporter_shows_reliability_and_flaky() -> None:
    console = Console(record=True, width=120)
    ConsoleReporter(console).render(_suite_result())
    out = console.export_text()
    assert "Reliability curve" in out
    assert "Flaky tasks" in out


def test_html_report_includes_reliability_and_flaky() -> None:
    html = render_html(_suite_result())
    assert "Reliability curve" in html
    assert "Flaky tasks" in html
    assert "split" in html
