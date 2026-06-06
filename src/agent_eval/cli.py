"""Command-line interface for the eval harness."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from agent_eval.graders import validate_suite_graders  # also registers graders
from agent_eval.harness import RunConfig, run_suite_to_disk
from agent_eval.reporters.console_reporter import ConsoleReporter
from agent_eval.reporters.html_reporter import HTMLReporter
from agent_eval.schemas import ScoringMode
from agent_eval.storage import load_results
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
    artifacts = run_suite_to_disk(loaded, output, config)

    ConsoleReporter(console).render(artifacts.result)
    console.print(
        f"\nWrote [cyan]{artifacts.json_path}[/cyan] and [cyan]{artifacts.html_path}[/cyan]."
    )


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
    results: Path = typer.Option(..., "--results", help="Path to a stored results.json."),
) -> None:
    """Browse stored results in an interactive terminal UI."""
    if not results.exists():
        console.print(f"[red]Results file not found:[/red] {results}")
        raise typer.Exit(code=1)
    from agent_eval.ui import run_ui

    try:
        run_ui(str(results))
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("textual"):
            raise
        console.print(
            "[red]The interactive UI requires the 'ui' extra.[/red] "
            "Install it with: [cyan]pip install 'agent-eval-harness\\[ui]'[/cyan]"
        )
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
