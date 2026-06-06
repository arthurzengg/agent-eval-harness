"""Pure presentation helpers for the TUI.

These functions turn result models into Rich console markup strings so they
can be unit-tested without instantiating a Textual app.
"""

from __future__ import annotations

import json

from agent_eval.schemas import (
    MetricsSummary,
    SuiteResult,
    TaskResult,
    TranscriptStep,
    TrialResult,
)

_ROLE_COLORS = {
    "user": "bright_white",
    "assistant": "dodger_blue1",
    "tool": "medium_purple1",
    "system": "grey62",
    "environment": "dark_sea_green4",
}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def trial_dots(task: TaskResult) -> str:
    """Compact per-trial pass/fail dots, e.g. ``●●○``."""
    return "".join("[green]●[/green]" if t.passed else "[red]●[/red]" for t in task.trials)


def task_label(task: TaskResult) -> str:
    """One-line task summary for the navigation tree."""
    return f"{task.task_id}  {trial_dots(task)}  {task.num_passed}/{task.num_trials}"


def trial_label(trial: TrialResult) -> str:
    """One-line trial summary for the navigation tree."""
    status = "[green]PASS[/green]" if trial.passed else "[red]FAIL[/red]"
    label = f"trial {trial.trial.index}  {status}  score {trial.score:.2f}"
    if trial.trial.latency_ms:
        label += f"  {trial.trial.latency_ms:.0f}ms"
    if trial.trial.error:
        label += "  [red]error[/red]"
    return label


def metrics_summary(metrics: MetricsSummary) -> str:
    """Suite-level metrics block shown in the sidebar."""
    return "\n".join(
        [
            f"[b]Tasks[/b]      {metrics.total_tasks}",
            f"[b]Trials[/b]     {metrics.total_trials}",
            f"[b]Pass rate[/b]  {_pct(metrics.pass_rate)}",
            f"[b]pass@{metrics.k}[/b]     {_pct(metrics.pass_at_k)}",
            f"[b]pass^{metrics.k}[/b]     {_pct(metrics.pass_caret_k)}",
            f"[b]Avg score[/b]  {metrics.avg_score:.3f}",
            f"[b]Latency[/b]    {metrics.avg_latency_ms:.0f}ms",
            f"[b]Errors[/b]     {_pct(metrics.error_rate)}",
        ]
    )


def _fmt_json(value: object) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


#: Maximum lines a JSON blob may occupy in the detail pane before truncation.
TRUNCATE_LINES = 12


def _truncate(text: str, expand: bool) -> str:
    """Cap ``text`` at ``TRUNCATE_LINES`` lines unless ``expand`` is set."""
    if expand:
        return text
    lines = text.splitlines()
    if len(lines) <= TRUNCATE_LINES:
        return text
    hidden = len(lines) - TRUNCATE_LINES
    kept = "\n".join(lines[:TRUNCATE_LINES])
    return f"{kept}\n[grey62]… (+{hidden} lines · press e to expand)[/grey62]"


def format_step(step: TranscriptStep, index: int, *, expand: bool = False) -> str:
    """Render one transcript step as markup lines."""
    color = _ROLE_COLORS.get(step.role.value, "white")
    lines = [f"[{color} b]{index:>3} {step.role.value.upper()}[/]"]
    if step.content is not None and step.content != "":
        content = step.content if isinstance(step.content, str) else _fmt_json(step.content)
        lines.append(f"    {_truncate(content, expand)}")
    if step.tool_call is not None:
        lines.append(f"    [medium_purple1]→ {step.tool_call.name}[/medium_purple1]")
        if step.tool_call.arguments:
            args = _truncate(_fmt_json(step.tool_call.arguments), expand)
            lines.append(f"      {args.replace(chr(10), chr(10) + '      ')}")
    if step.tool_result is not None:
        name = step.tool_result.name or "tool"
        if step.tool_result.error:
            lines.append(f"    [red]← {name} error: {step.tool_result.error}[/red]")
        else:
            content = _truncate(_fmt_json(step.tool_result.content), expand)
            content = content.replace("\n", "\n      ")
            lines.append(f"    [dark_sea_green4]← {name}[/dark_sea_green4] {content}")
    meta: list[str] = []
    if step.duration_ms is not None:
        meta.append(f"{step.duration_ms:.0f}ms")
    if step.token_usage is not None:
        meta.append(f"{step.token_usage.total_tokens} tok")
    if step.error:
        meta.append(f"[red]error: {step.error}[/red]")
    if meta:
        lines.append(f"    [grey62]{' · '.join(meta)}[/grey62]")
    return "\n".join(lines)


def format_graders(trial: TrialResult) -> str:
    """Render the grader verdicts for a trial."""
    if not trial.grader_results:
        return "[grey62]No graders ran for this trial.[/grey62]"
    lines = []
    for gr in trial.grader_results:
        mark = "[green]✓[/green]" if gr.passed else "[red]✗[/red]"
        line = f"{mark} {gr.grader_type:<22} {gr.score:.2f}  (w={gr.weight:g})"
        if not gr.enabled:
            line += "  [grey62]disabled[/grey62]"
        if gr.hard_fail:
            line += "  [red]hard fail[/red]"
        lines.append(line)
        if gr.reason and not gr.passed:
            lines.append(f"    [red]{gr.reason}[/red]")
    return "\n".join(lines)


def format_trial_detail(trial: TrialResult, *, expand: bool = False) -> str:
    """Full right-pane detail for a trial: header, graders, transcript."""
    status = "[green b]PASS[/]" if trial.passed else "[red b]FAIL[/]"
    parts = [
        f"{status}  score [b]{trial.score:.2f}[/b]"
        f"  ·  latency {trial.trial.latency_ms:.0f}ms"
        f"  ·  {len(trial.trial.transcript.steps)} steps"
        f"  ·  {len(trial.trial.transcript.tool_calls())} tool calls",
    ]
    if trial.trial.error:
        parts.append(f"[red]Trial error: {trial.trial.error}[/red]")
    parts.append("\n[b]Graders[/b]")
    parts.append(format_graders(trial))
    if trial.trial.outcome.state:
        parts.append("\n[b]Outcome[/b]")
        parts.append(_truncate(_fmt_json(trial.trial.outcome.state), expand))
    if trial.trial.final_output:
        parts.append("\n[b]Final output[/b]")
        parts.append(trial.trial.final_output)
    parts.append("\n[b]Transcript[/b]")
    parts.extend(
        format_step(step, i, expand=expand) for i, step in enumerate(trial.trial.transcript.steps)
    )
    return "\n".join(parts)


def live_trial_label(index: int, status: str, result: TrialResult | None = None) -> str:
    """One-line label for a trial in the live-run tree.

    ``status`` is one of ``pending`` / ``running`` / ``done``; for ``done`` the
    finished ``result`` supplies the pass/fail verdict and score.
    """
    if status == "running":
        return f"trial {index}  [yellow]▶ running[/yellow]"
    if status == "done" and result is not None:
        return trial_label(result)
    return f"trial {index}  [grey62]· pending[/grey62]"


def live_progress(done: int, total: int, passed: int, elapsed_s: float) -> str:
    """Sidebar progress block shown while a live run is in flight."""
    failed = done - passed
    return "\n".join(
        [
            f"[b]Progress[/b]   {done}/{total} trials",
            f"[b]Passed[/b]     [green]{passed}[/green]",
            f"[b]Failed[/b]     [red]{failed}[/red]" if failed else "[b]Failed[/b]     0",
            f"[b]Elapsed[/b]    {elapsed_s:.1f}s",
        ]
    )


def suite_title(result: SuiteResult) -> str:
    """Window/header title for the browser."""
    return f"{result.suite.name} ({result.suite.id} v{result.suite.version})"
