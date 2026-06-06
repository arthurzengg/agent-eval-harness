"""Tests for the interactive results browser (presenter + app smoke test)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.schemas import (
    GraderResult,
    MetricsSummary,
    Role,
    SuiteMetadata,
    SuiteResult,
    TaskResult,
    ToolCall,
    ToolResult,
    Transcript,
    TranscriptStep,
    Trial,
    TrialResult,
)
from agent_eval.storage import write_results
from agent_eval.ui import presenter


def _trial_result(*, passed: bool, index: int = 0) -> TrialResult:
    transcript = Transcript(
        steps=[
            TranscriptStep(role=Role.user, content="I want a refund."),
            TranscriptStep(
                role=Role.assistant,
                content="Checking.",
                tool_call=ToolCall(name="verify_identity", arguments={"user": "a"}),
            ),
            TranscriptStep(
                role=Role.tool,
                tool_result=ToolResult(name="verify_identity", content={"ok": True}),
            ),
        ]
    )
    return TrialResult(
        trial=Trial(task_id="t", index=index, transcript=transcript, latency_ms=42.0),
        grader_results=[
            GraderResult(grader_type="required_tools", score=1.0, passed=True),
            GraderResult(
                grader_type="tool_sequence", score=0.4, passed=False, reason="missing step"
            ),
        ],
        score=0.7,
        passed=passed,
    )


def test_trial_dots_and_task_label() -> None:
    task = TaskResult(
        task_id="refund",
        trials=[_trial_result(passed=True), _trial_result(passed=False, index=1)],
        pass_rate=0.5,
        avg_score=0.7,
    )
    dots = presenter.trial_dots(task)
    assert dots.count("[green]●[/green]") == 1
    assert dots.count("[red]●[/red]") == 1
    assert "refund" in presenter.task_label(task)
    assert "1/2" in presenter.task_label(task)


def test_trial_label_shows_status_and_latency() -> None:
    label = presenter.trial_label(_trial_result(passed=True))
    assert "PASS" in label
    assert "score 0.70" in label
    assert "42ms" in label


def test_metrics_summary_includes_pass_at_k() -> None:
    text = presenter.metrics_summary(
        MetricsSummary(total_tasks=2, total_trials=6, pass_at_k=1.0, pass_caret_k=0.5, k=3)
    )
    assert "pass@3" in text
    assert "pass^3" in text
    assert "100.0%" in text


def test_format_trial_detail_covers_graders_and_transcript() -> None:
    detail = presenter.format_trial_detail(_trial_result(passed=False))
    assert "FAIL" in detail
    assert "required_tools" in detail
    assert "missing step" in detail  # failing grader reason surfaced
    assert "→ verify_identity" in detail  # tool call rendered
    assert "← verify_identity" in detail  # tool result rendered


def test_format_step_handles_errors() -> None:
    step = TranscriptStep(
        role=Role.tool,
        tool_result=ToolResult(name="boom", error="exploded"),
        error="step failed",
    )
    text = presenter.format_step(step, 0)
    assert "boom error: exploded" in text
    assert "step failed" in text


async def test_app_smoke_navigates_results(tmp_path: Path) -> None:
    textual = pytest.importorskip("textual")
    assert textual is not None
    from agent_eval.ui.app import EvalBrowserApp

    result = SuiteResult(
        suite=SuiteMetadata(id="demo", name="Demo Suite"),
        task_results=[
            TaskResult(
                task_id="t",
                trials=[_trial_result(passed=True), _trial_result(passed=False, index=1)],
                pass_rate=0.5,
                avg_score=0.7,
            )
        ],
        metrics=MetricsSummary(total_tasks=1, total_trials=2, k=2),
    )
    results_path = write_results(result, tmp_path)

    app = EvalBrowserApp(results_path)
    async with app.run_test(size=(120, 40)) as pilot:
        # Detail pane is populated on mount (first trial auto-selected).
        detail = app.query_one("#detail")
        assert str(detail.render())  # non-empty
        await pilot.press("down", "down", "enter")
        assert str(detail.render())
        await pilot.press("q")
