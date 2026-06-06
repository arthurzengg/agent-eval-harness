"""Tests for the statistical confidence and significance module."""

from __future__ import annotations

from agent_eval.compare import significance_report
from agent_eval.schemas import MetricsSummary, SuiteMetadata, SuiteResult, TaskResult
from agent_eval.stats import (
    bootstrap_ci,
    cohens_d,
    compare_significance,
    paired_bootstrap,
    paired_t_test,
)


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    ci = bootstrap_ci(values, iterations=500, seed=1)
    assert ci.low <= ci.point <= ci.high
    assert 0.0 <= ci.low <= 1.0
    assert 0.0 <= ci.high <= 1.0
    assert ci.margin >= 0.0


def test_bootstrap_ci_is_deterministic() -> None:
    values = [0.2, 0.4, 0.6, 0.8]
    a = bootstrap_ci(values, iterations=300, seed=7)
    b = bootstrap_ci(values, iterations=300, seed=7)
    assert (a.low, a.point, a.high) == (b.low, b.point, b.high)


def test_bootstrap_ci_single_point_collapses() -> None:
    ci = bootstrap_ci([0.5])
    assert ci.low == ci.point == ci.high == 0.5


def test_paired_bootstrap_detects_clear_improvement() -> None:
    baseline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    current = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    res = paired_bootstrap(baseline, current, iterations=500, seed=2)
    assert res.improved
    assert res.mean_delta == 1.0
    assert res.p_value < 0.05


def test_paired_bootstrap_no_difference_is_not_significant() -> None:
    baseline = [0.5, 0.5, 0.5, 0.5]
    current = [0.5, 0.5, 0.5, 0.5]
    res = paired_bootstrap(baseline, current, iterations=300, seed=3)
    assert res.mean_delta == 0.0
    assert res.p_value == 1.0


def test_paired_bootstrap_length_mismatch_raises() -> None:
    try:
        paired_bootstrap([0.1], [0.1, 0.2])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError")


def test_cohens_d_zero_when_no_variation() -> None:
    assert cohens_d([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_cohens_d_positive_for_consistent_gain() -> None:
    d = cohens_d([0.0, 0.1, 0.2], [0.5, 0.6, 0.7])
    assert d > 0


def test_paired_t_test_clear_difference_small_p() -> None:
    baseline = [0.1, 0.2, 0.15, 0.05, 0.1]
    current = [0.8, 0.9, 0.85, 0.95, 0.9]
    res = paired_t_test(baseline, current)
    assert res.t > 0
    assert res.df == 4
    assert res.p_value < 0.01


def test_paired_t_test_identical_is_neutral() -> None:
    res = paired_t_test([0.4, 0.6], [0.4, 0.6])
    assert res.t == 0.0
    assert res.p_value == 1.0


def test_paired_t_test_single_pair_is_neutral() -> None:
    res = paired_t_test([0.4], [0.9])
    assert res.p_value == 1.0


def test_compare_significance_flags_regression() -> None:
    baseline = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    report = compare_significance(baseline, current, metric="pass_rate", iterations=500)
    assert report.significant
    assert report.significant_regression
    assert "pass_rate" in report.summary()


def test_compare_significance_noise_is_not_significant() -> None:
    baseline = [0.5, 0.6, 0.4, 0.55, 0.45]
    current = [0.52, 0.58, 0.42, 0.54, 0.46]
    report = compare_significance(baseline, current, iterations=500)
    assert not report.significant_regression


def _suite(pass_rates: dict[str, float]) -> SuiteResult:
    return SuiteResult(
        suite=SuiteMetadata(id="s", name="S"),
        task_results=[TaskResult(task_id=tid, k=1, pass_rate=pr) for tid, pr in pass_rates.items()],
        metrics=MetricsSummary(total_tasks=len(pass_rates)),
    )


def test_significance_report_over_suite_results() -> None:
    base = _suite({"t1": 1.0, "t2": 1.0, "t3": 1.0, "t4": 1.0})
    cur = _suite({"t1": 0.0, "t2": 0.0, "t3": 0.0, "t4": 0.0})
    report = significance_report(base, cur, alpha=0.05)
    assert report.n_pairs == 4
    assert report.significant_regression
