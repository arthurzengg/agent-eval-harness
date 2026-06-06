"""Compare a current suite run against a baseline to gate regressions.

The comparison is pure (no I/O): given two ``SuiteResult`` objects it reports
per-metric deltas and per-task pass-rate changes, and whether anything regressed
beyond a tolerance. The CLI turns ``regressed`` into a non-zero exit code so a
run can fail CI when quality drops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_eval.schemas import SuiteResult
from agent_eval.stats import SignificanceReport, compare_significance

# Metrics where a higher value is better; these gate the comparison.
GATED_METRICS = ("pass_rate", "pass_at_k", "pass_caret_k", "avg_score")


@dataclass
class MetricDelta:
    """Change in one suite-level metric (higher is better)."""

    name: str
    baseline: float
    current: float
    regressed: bool

    @property
    def delta(self) -> float:
        return self.current - self.baseline


@dataclass
class TaskDelta:
    """Change in one task's pass rate between baseline and current."""

    task_id: str
    status: str  # "both", "new", "removed"
    baseline: float
    current: float
    regressed: bool

    @property
    def delta(self) -> float:
        return self.current - self.baseline


@dataclass
class ComparisonReport:
    """The full baseline-vs-current comparison."""

    tolerance: float
    metrics: list[MetricDelta] = field(default_factory=list)
    tasks: list[TaskDelta] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        """True if any gated metric or any shared task regressed."""
        return any(m.regressed for m in self.metrics) or any(t.regressed for t in self.tasks)

    @property
    def regressions(self) -> list[str]:
        """Human-readable lines for each regression found."""
        lines = [
            f"{m.name}: {m.baseline:.3f} -> {m.current:.3f} ({m.delta:+.3f})"
            for m in self.metrics
            if m.regressed
        ]
        lines += [
            f"task '{t.task_id}': {t.baseline:.3f} -> {t.current:.3f} ({t.delta:+.3f})"
            for t in self.tasks
            if t.regressed
        ]
        return lines


def compare_results(
    baseline: SuiteResult, current: SuiteResult, tolerance: float = 0.0
) -> ComparisonReport:
    """Compare ``current`` against ``baseline``.

    A gated metric or a shared task regresses when its value drops by more than
    ``tolerance``. New tasks (only in current) and removed tasks (only in
    baseline) are reported for context but never regress the gate on their own.
    """
    report = ComparisonReport(tolerance=tolerance)

    for name in GATED_METRICS:
        base_val = float(getattr(baseline.metrics, name))
        cur_val = float(getattr(current.metrics, name))
        report.metrics.append(
            MetricDelta(
                name=name,
                baseline=base_val,
                current=cur_val,
                regressed=cur_val < base_val - tolerance,
            )
        )

    base_tasks = {tr.task_id: tr.pass_rate for tr in baseline.task_results}
    cur_tasks = {tr.task_id: tr.pass_rate for tr in current.task_results}
    for task_id in sorted(base_tasks.keys() | cur_tasks.keys()):
        in_base = task_id in base_tasks
        in_cur = task_id in cur_tasks
        base_val = base_tasks.get(task_id, 0.0)
        cur_val = cur_tasks.get(task_id, 0.0)
        status = "both" if in_base and in_cur else ("removed" if in_base else "new")
        # Only shared tasks gate; missing on either side is informational.
        regressed = status == "both" and cur_val < base_val - tolerance
        report.tasks.append(
            TaskDelta(
                task_id=task_id,
                status=status,
                baseline=base_val,
                current=cur_val,
                regressed=regressed,
            )
        )

    return report


def shared_task_pass_rates(
    baseline: SuiteResult, current: SuiteResult
) -> tuple[list[float], list[float]]:
    """Aligned per-task pass rates for tasks present in both runs.

    Returns ``(baseline_values, current_values)`` in a stable, shared task order.
    Tasks present on only one side are dropped, since a paired test needs pairs.
    """
    base = {tr.task_id: tr.pass_rate for tr in baseline.task_results}
    cur = {tr.task_id: tr.pass_rate for tr in current.task_results}
    shared = sorted(base.keys() & cur.keys())
    return [base[t] for t in shared], [cur[t] for t in shared]


def significance_report(
    baseline: SuiteResult,
    current: SuiteResult,
    *,
    metric: str = "pass_rate",
    alpha: float = 0.05,
) -> SignificanceReport:
    """Paired statistical comparison of per-task pass rates between two runs.

    Use ``SignificanceReport.significant_regression`` to gate CI on a
    statistically significant drop rather than any raw decrease.
    """
    base_vals, cur_vals = shared_task_pass_rates(baseline, current)
    return compare_significance(base_vals, cur_vals, metric=metric, alpha=alpha)
