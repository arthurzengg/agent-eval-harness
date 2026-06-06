"""Console reporter using Rich."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent_eval.reliability import flaky_tasks, suite_reliability_curve
from agent_eval.schemas import SuiteResult
from agent_eval.taxonomy import aggregate_failures


class ConsoleReporter:
    """Prints a concise terminal summary of a suite run."""

    name = "console"

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def render(self, result: SuiteResult, output_dir: Path | None = None) -> None:
        m = result.metrics
        c = self._console
        c.rule(f"[bold]{result.suite.name}[/bold]  ({result.suite.id})")

        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_row("Tasks", str(m.total_tasks))
        table.add_row("Trials", str(m.total_trials))
        table.add_row("Pass rate (per trial)", _pct(m.pass_rate))
        kl = m.k_label()
        table.add_row(f"pass@{kl}", _pct(m.pass_at_k))
        table.add_row(f"pass^{kl}", _pct(m.pass_caret_k))
        if not m.consistent_k:
            table.add_row(
                "Trials per task",
                f"[yellow]mixed (k={m.k_min}..{m.k_max}); pass@k/pass^k are per-task[/yellow]",
            )
        table.add_row("Avg score", f"{m.avg_score:.3f}")
        table.add_row("Avg latency (ms)", f"{m.avg_latency_ms:.1f}")
        table.add_row("Error rate", _pct(m.error_rate))
        if m.total_tokens:
            table.add_row(
                "Avg tokens (in/out)",
                f"{m.avg_input_tokens:.0f} / {m.avg_output_tokens:.0f}",
            )
            table.add_row("Total tokens", f"{m.total_tokens:,}")
        if m.total_cost_usd:
            table.add_row("Avg cost / trial", f"${m.avg_cost_usd:.4f}")
            table.add_row("Total cost", f"${m.total_cost_usd:.4f}")
        c.print(table)

        self._reliability(result)
        self._failed_tasks(result)
        self._taxonomy(result)
        self._top_failures(result)

    def _taxonomy(self, result: SuiteResult) -> None:
        agg = aggregate_failures({}, result.task_results)
        if not agg.total_failures:
            return
        table = Table(title="Failure taxonomy")
        table.add_column("Category")
        table.add_column("Count", justify="right")
        for category, count in agg.most_common():
            table.add_row(str(category), str(count))
        self._console.print(table)

    def _reliability(self, result: SuiteResult) -> None:
        curve = suite_reliability_curve(result.task_results)
        # Only worth showing once there is more than one attempt to scale over.
        if len(curve) < 2:
            return
        table = Table(title="Reliability curve")
        table.add_column("k", justify="right")
        table.add_column("pass@k", justify="right")
        table.add_column("pass^k", justify="right")
        table.add_column("tasks", justify="right")
        for p in curve:
            table.add_row(str(p.k), _pct(p.pass_at_k), _pct(p.pass_caret_k), str(p.n_tasks))
        self._console.print(table)

        flaky = flaky_tasks(result.task_results)
        if flaky:
            ftable = Table(title="Flaky tasks (pass inconsistently across trials)")
            ftable.add_column("Task")
            ftable.add_column("Passed / Trials")
            ftable.add_column("Flakiness", justify="right")
            for f in flaky:
                ftable.add_row(f.task_id, f"{f.n_passed} / {f.n_trials}", f"{f.flakiness:.2f}")
            self._console.print(ftable)

    def _failed_tasks(self, result: SuiteResult) -> None:
        failed = [tr for tr in result.task_results if not all(t.passed for t in tr.trials)]
        if not failed:
            self._console.print("[green]All tasks passed every trial.[/green]")
            return
        table = Table(title="Tasks with failing trials")
        table.add_column("Task")
        table.add_column("Passed / Trials")
        table.add_column("Avg score")
        for tr in failed:
            table.add_row(tr.task_id, f"{tr.num_passed} / {tr.num_trials}", f"{tr.avg_score:.3f}")
        self._console.print(table)

    def _top_failures(self, result: SuiteResult) -> None:
        reasons: Counter[str] = Counter()
        for tr in result.task_results:
            for trial in tr.trials:
                for gr in trial.grader_results:
                    if not gr.passed and gr.enabled and gr.reason:
                        reasons[f"[{gr.grader_type}] {gr.reason}"] += 1
        if not reasons:
            return
        table = Table(title="Top failure reasons")
        table.add_column("Count", justify="right")
        table.add_column("Reason")
        for reason, count in reasons.most_common(5):
            table.add_row(str(count), reason)
        self._console.print(table)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"
