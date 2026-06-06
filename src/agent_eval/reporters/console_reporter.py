"""Console reporter using Rich."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent_eval.schemas import SuiteResult


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
        table.add_row(f"pass@{m.k}", _pct(m.pass_at_k))
        table.add_row(f"pass^{m.k}", _pct(m.pass_caret_k))
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

        self._failed_tasks(result)
        self._top_failures(result)

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
