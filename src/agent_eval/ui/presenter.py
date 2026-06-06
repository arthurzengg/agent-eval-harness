"""Pure presentation helpers for the TUI.

These functions turn result models into Rich console markup strings so they
can be unit-tested without instantiating a Textual app.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime

from agent_eval.compare import ComparisonReport, TaskDelta
from agent_eval.schemas import (
    MetricsSummary,
    SuiteResult,
    TaskResult,
    TranscriptStep,
    TrialResult,
)
from agent_eval.storage import RunInfo

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


def run_label(run: RunInfo) -> str:
    """One-line summary of a stored run for the run picker."""
    when = datetime.fromtimestamp(run.mtime).strftime("%Y-%m-%d %H:%M")
    color = "green" if run.pass_rate >= 1.0 else "yellow" if run.pass_rate > 0.0 else "red"
    return (
        f"[b]{run.suite_name}[/b]  [grey62]{when}[/grey62]  "
        f"[{color}]{_pct(run.pass_rate)}[/{color}] of {run.total_trials} trials  "
        f"[grey62]{run.path}[/grey62]"
    )


def _delta_markup(delta: float, *, as_pct: bool = True) -> str:
    """Render a signed delta with a colored direction arrow."""
    value = f"{delta * 100:+.1f}%" if as_pct else f"{delta:+.3f}"
    if delta > 0:
        return f"[green]▲ {value}[/green]"
    if delta < 0:
        return f"[red]▼ {value}[/red]"
    return f"[grey62]= {value}[/grey62]"


def compare_summary(report: ComparisonReport, baseline: SuiteResult, current: SuiteResult) -> str:
    """Sidebar block: suite-level metric deltas between two runs."""
    lines = ["[b]Metric deltas[/b]"]
    for m in report.metrics:
        lines.append(f"{m.name:<12} {m.baseline:.3f} → {m.current:.3f}  {_delta_markup(m.delta)}")
    lat_delta = current.metrics.avg_latency_ms - baseline.metrics.avg_latency_ms
    lat_mark = "[red]" if lat_delta > 0 else "[green]" if lat_delta < 0 else "[grey62]"
    lines.append(
        f"{'latency':<12} {baseline.metrics.avg_latency_ms:.0f}ms → "
        f"{current.metrics.avg_latency_ms:.0f}ms  {lat_mark}{lat_delta:+.0f}ms[/]"
    )
    verdict = (
        "[red b]REGRESSED[/red b]" if report.regressed else "[green b]NO REGRESSIONS[/green b]"
    )
    lines.append(f"\n{verdict}  [grey62](tolerance {report.tolerance:g})[/grey62]")
    return "\n".join(lines)


def compare_task_label(delta: TaskDelta) -> str:
    """One-line task summary for the comparison tree."""
    if delta.status == "new":
        return f"{delta.task_id}  [cyan]new[/cyan]  {_pct(delta.current)}"
    if delta.status == "removed":
        return f"{delta.task_id}  [grey62]removed[/grey62]  was {_pct(delta.baseline)}"
    return (
        f"{delta.task_id}  {_pct(delta.baseline)} → {_pct(delta.current)}  "
        f"{_delta_markup(delta.delta)}"
    )


def _tool_sequence(trial: TrialResult) -> list[str]:
    return [call.name for call in trial.trial.transcript.tool_calls()]


def tool_sequence_diff(baseline: list[str], current: list[str]) -> str:
    """Unified-style diff of two tool-call name sequences."""
    if baseline == current:
        return "[grey62]identical tool sequences[/grey62]"
    lines = []
    for line in difflib.ndiff(baseline, current):
        if line.startswith("- "):
            lines.append(f"[red]- {line[2:]}[/red]")
        elif line.startswith("+ "):
            lines.append(f"[green]+ {line[2:]}[/green]")
        elif line.startswith("  "):
            lines.append(f"[grey62]  {line[2:]}[/grey62]")
        # "? " hint lines add noise for name lists; skip them.
    return "\n".join(lines)


def format_task_compare(
    delta: TaskDelta, baseline: TaskResult | None, current: TaskResult | None
) -> str:
    """Right-pane detail for one task in the comparison view."""
    parts = [f"[b]{delta.task_id}[/b]"]
    if delta.status == "new":
        parts.append("[cyan]Only in the current run.[/cyan]")
    elif delta.status == "removed":
        parts.append("[grey62]Only in the baseline run.[/grey62]")
    parts.append(
        f"pass rate {_pct(delta.baseline)} → {_pct(delta.current)}  {_delta_markup(delta.delta)}"
    )
    base_trials = baseline.trials if baseline else []
    cur_trials = current.trials if current else []
    if base_trials or cur_trials:
        parts.append(
            f"\n[b]Trials[/b]  baseline {''.join(_dot(t) for t in base_trials) or '—'}"
            f"  ·  current {''.join(_dot(t) for t in cur_trials) or '—'}"
        )
    for index in range(max(len(base_trials), len(cur_trials))):
        base_seq = _tool_sequence(base_trials[index]) if index < len(base_trials) else []
        cur_seq = _tool_sequence(cur_trials[index]) if index < len(cur_trials) else []
        parts.append(f"\n[b]Trial {index} tool calls[/b]  (baseline vs current)")
        parts.append(tool_sequence_diff(base_seq, cur_seq))
    return "\n".join(parts)


def _dot(trial: TrialResult) -> str:
    return "[green]●[/green]" if trial.passed else "[red]●[/red]"
