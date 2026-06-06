from agent_eval.compare import compare_results
from agent_eval.schemas import MetricsSummary, SuiteMetadata, SuiteResult, TaskResult


def _result(pass_rate: float, avg_score: float, tasks: dict[str, float]) -> SuiteResult:
    return SuiteResult(
        suite=SuiteMetadata(id="s", name="S"),
        task_results=[TaskResult(task_id=k, pass_rate=v) for k, v in tasks.items()],
        metrics=MetricsSummary(
            pass_rate=pass_rate,
            pass_at_k=pass_rate,
            pass_caret_k=pass_rate,
            avg_score=avg_score,
        ),
    )


def test_no_regression_when_improved() -> None:
    base = _result(0.8, 0.8, {"a": 1.0, "b": 0.5})
    cur = _result(0.9, 0.9, {"a": 1.0, "b": 1.0})
    report = compare_results(base, cur)
    assert not report.regressed
    assert report.regressions == []


def test_regression_on_metric_drop() -> None:
    base = _result(0.9, 0.9, {"a": 1.0})
    cur = _result(0.7, 0.9, {"a": 1.0})  # pass_rate drops
    report = compare_results(base, cur)
    assert report.regressed
    assert any("pass_rate" in line for line in report.regressions)


def test_within_tolerance_is_not_a_regression() -> None:
    base = _result(0.90, 0.90, {"a": 1.0})
    cur = _result(0.85, 0.90, {"a": 1.0})  # 0.05 drop
    assert compare_results(base, cur, tolerance=0.05).regressed is False
    assert compare_results(base, cur, tolerance=0.04).regressed is True


def test_per_task_regression_detected() -> None:
    base = _result(0.9, 0.9, {"a": 1.0, "b": 1.0})
    cur = _result(0.9, 0.9, {"a": 1.0, "b": 0.0})  # task b regressed
    report = compare_results(base, cur)
    assert report.regressed
    assert any("task 'b'" in line for line in report.regressions)


def test_new_and_removed_tasks_do_not_gate() -> None:
    base = _result(0.9, 0.9, {"a": 1.0, "removed": 1.0})
    cur = _result(0.9, 0.9, {"a": 1.0, "added": 0.0})
    report = compare_results(base, cur)
    statuses = {t.task_id: t.status for t in report.tasks}
    assert statuses["removed"] == "removed"
    assert statuses["added"] == "new"
    assert not report.regressed  # neither side-only task fails the gate
