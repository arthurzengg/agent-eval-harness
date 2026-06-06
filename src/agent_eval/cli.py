"""Command-line interface for the eval harness."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agent_eval.compare import compare_results
from agent_eval.graders import validate_suite_graders  # also registers graders
from agent_eval.harness import RunConfig, run_suite_to_disk, write_reports
from agent_eval.reporters.console_reporter import ConsoleReporter
from agent_eval.reporters.html_reporter import HTMLReporter
from agent_eval.schemas import EvalSuite, ScoringMode
from agent_eval.storage import discover_runs, load_results
from agent_eval.suite_loader import SuiteLoadError, load_suite

app = typer.Typer(
    add_completion=False,
    help="Run automated evaluations for tool-calling / agentic systems.",
)
console = Console()


@app.command()
def validate(suite: Path = typer.Argument(..., help="Path to an eval suite YAML file.")) -> None:
    """Validate an eval suite file."""
    try:
        loaded = load_suite(suite)
    except SuiteLoadError as exc:
        console.print(f"[red]Invalid suite:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    grader_errors = validate_suite_graders(loaded)
    if grader_errors:
        console.print(f"[red]Invalid grader configuration in {suite}:[/red]")
        for err in grader_errors:
            console.print(f"  - {err}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]OK[/green] {suite}: suite '{loaded.suite.id}' with {len(loaded.tasks)} task(s)."
    )


@app.command()
def run(
    suite: Path = typer.Option(..., "--suite", help="Path to an eval suite YAML file."),
    agent: str = typer.Option("echo", "--agent", help="Agent adapter name (echo, http)."),
    output: Path = typer.Option(..., "--output", help="Output directory for reports."),
    trials: int | None = typer.Option(None, "--trials", help="Override the trial count."),
    agent_url: str = typer.Option("", "--agent-url", help="URL for the http agent adapter."),
    keep_workdirs: bool = typer.Option(False, "--keep-workdirs", help="Keep trial temp dirs."),
    scoring_mode: str | None = typer.Option(
        None, "--scoring-mode", help="Override scoring mode (weighted | binary)."
    ),
    concurrency: int = typer.Option(
        1, "--concurrency", min=1, help="Max trials to run at once (default 1 = serial)."
    ),
    ui: bool = typer.Option(
        False, "--ui", help="Watch the run live in the terminal UI, then browse results."
    ),
) -> None:
    """Run an eval suite against an agent adapter."""
    try:
        loaded = load_suite(suite)
    except SuiteLoadError as exc:
        console.print(f"[red]Invalid suite:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    config = RunConfig(
        agent=agent,
        agent_url=agent_url,
        trials=trials,
        scoring_mode=ScoringMode(scoring_mode) if scoring_mode is not None else None,
        concurrency=concurrency,
        keep_workdirs=keep_workdirs,
    )

    if ui:
        _run_with_ui(loaded, output, config)
        return

    artifacts = run_suite_to_disk(loaded, output, config)

    ConsoleReporter(console).render(artifacts.result)
    console.print(
        f"\nWrote [cyan]{artifacts.json_path}[/cyan] and [cyan]{artifacts.html_path}[/cyan]."
    )


def _run_with_ui(loaded: EvalSuite, output: Path, config: RunConfig) -> None:
    """Run the suite under the live TUI, persist reports, open the browser."""
    from agent_eval.ui import run_live, run_ui

    try:
        result = run_live(loaded, config)
    except ModuleNotFoundError as exc:
        _require_ui_extra(exc)
        raise  # unreachable; _require_ui_extra always raises

    if result is None:
        console.print("[yellow]Run aborted; no results written.[/yellow]")
        raise typer.Exit(code=1)

    artifacts = write_reports(result, output)
    console.print(
        f"Wrote [cyan]{artifacts.json_path}[/cyan] and [cyan]{artifacts.html_path}[/cyan]."
    )
    if artifacts.json_path is not None:
        run_ui(str(artifacts.json_path))


def _require_ui_extra(exc: ModuleNotFoundError) -> None:
    """Translate a missing-textual import into a friendly install hint."""
    if exc.name is None or not exc.name.startswith("textual"):
        raise exc
    console.print(
        "[red]The interactive UI requires the 'ui' extra.[/red] "
        "Install it with: [cyan]pip install 'agent-eval-harness\\[ui]'[/cyan]"
    )
    raise typer.Exit(code=1) from exc


@app.command()
def report(
    results: Path = typer.Option(..., "--results", help="Path to a stored results.json."),
    output: Path = typer.Option(..., "--output", help="Path for the generated HTML file."),
) -> None:
    """Regenerate an HTML report from stored JSON results."""
    if not results.exists():
        console.print(f"[red]Results file not found:[/red] {results}")
        raise typer.Exit(code=1)
    result = load_results(results)
    path = HTMLReporter().render_to_file(result, output)
    console.print(f"[green]Wrote[/green] {path}")


@app.command()
def ui(
    results: Path | None = typer.Option(
        None, "--results", help="Path to a stored results.json. Omit to pick from --dir."
    ),
    directory: Path = typer.Option(
        Path("reports"),
        "--dir",
        help="Directory to scan for stored runs when --results is omitted.",
    ),
    compare: tuple[Path, Path] | None = typer.Option(
        None,
        "--compare",
        help="Two results.json paths (baseline current) to diff side by side.",
    ),
) -> None:
    """Browse stored results in an interactive terminal UI."""
    if compare is not None:
        _compare_ui(*compare)
        return
    if results is None:
        results = _pick_results(directory)
    if not results.exists():
        console.print(f"[red]Results file not found:[/red] {results}")
        raise typer.Exit(code=1)
    from agent_eval.ui import run_ui

    try:
        run_ui(str(results))
    except ModuleNotFoundError as exc:
        _require_ui_extra(exc)


def _compare_ui(baseline: Path, current: Path) -> None:
    """Open the run-comparison browser over two stored results files."""
    for label, path in (("Baseline", baseline), ("Current", current)):
        if not path.exists():
            console.print(f"[red]{label} results file not found:[/red] {path}")
            raise typer.Exit(code=1)
    from agent_eval.ui import run_compare_ui

    try:
        run_compare_ui(str(baseline), str(current))
    except ModuleNotFoundError as exc:
        _require_ui_extra(exc)


def _pick_results(directory: Path) -> Path:
    """Scan ``directory`` for stored runs and let the user pick one."""
    runs = discover_runs(directory) if directory.exists() else []
    if not runs:
        console.print(
            f"[red]No stored runs found under[/red] {directory}[red].[/red] "
            "Run a suite first, or pass [cyan]--results[/cyan] / [cyan]--dir[/cyan]."
        )
        raise typer.Exit(code=1)

    from agent_eval.ui import pick_run

    try:
        chosen = pick_run(runs)
    except ModuleNotFoundError as exc:
        _require_ui_extra(exc)
        raise  # unreachable; _require_ui_extra always raises
    if chosen is None:
        raise typer.Exit(code=0)  # user quit the picker
    return chosen


@app.command()
def compare(
    baseline: Path = typer.Option(..., "--baseline", help="Path to the baseline results.json."),
    current: Path = typer.Option(..., "--current", help="Path to the current results.json."),
    tolerance: float = typer.Option(
        0.0, "--tolerance", min=0.0, help="Allowed drop before a metric counts as regressed."
    ),
) -> None:
    """Compare a run against a baseline; exit non-zero on regression."""
    for label, path in (("Baseline", baseline), ("Current", current)):
        if not path.exists():
            console.print(f"[red]{label} results file not found:[/red] {path}")
            raise typer.Exit(code=1)

    report = compare_results(load_results(baseline), load_results(current), tolerance)

    table = Table(title=f"Baseline vs current (tolerance {tolerance:g})")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    for m in report.metrics:
        color = "red" if m.regressed else ("green" if m.delta > 0 else "white")
        table.add_row(
            m.name,
            f"{m.baseline:.3f}",
            f"{m.current:.3f}",
            f"[{color}]{m.delta:+.3f}[/{color}]",
        )
    console.print(table)

    regressed_tasks = [t for t in report.tasks if t.regressed]
    if regressed_tasks:
        ttable = Table(title="Regressed tasks")
        ttable.add_column("Task")
        ttable.add_column("Baseline", justify="right")
        ttable.add_column("Current", justify="right")
        for t in regressed_tasks:
            ttable.add_row(t.task_id, f"{t.baseline:.3f}", f"{t.current:.3f}")
        console.print(ttable)

    if report.regressed:
        console.print(f"[red]REGRESSED[/red] ({len(report.regressions)} check(s) dropped).")
        raise typer.Exit(code=1)
    console.print("[green]OK[/green] no regressions beyond tolerance.")


if __name__ == "__main__":
    app()
