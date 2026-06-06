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


def _trial_result(*, passed: bool, index: int = 0, tool: str = "verify_identity") -> TrialResult:
    transcript = Transcript(
        steps=[
            TranscriptStep(role=Role.user, content="I want a refund."),
            TranscriptStep(
                role=Role.assistant,
                content="Checking.",
                tool_call=ToolCall(name=tool, arguments={"user": "a"}),
            ),
            TranscriptStep(
                role=Role.tool,
                tool_result=ToolResult(name=tool, content={"ok": True}),
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


def test_truncate_caps_long_blobs() -> None:
    big = {f"key_{i}": i for i in range(50)}
    step = TranscriptStep(role=Role.tool, tool_result=ToolResult(name="big", content=big))
    short = presenter.format_step(step, 0)
    assert "press e to expand" in short
    full = presenter.format_step(step, 0, expand=True)
    assert "press e to expand" not in full
    assert "key_49" in full


def test_truncate_leaves_short_blobs_alone() -> None:
    step = TranscriptStep(
        role=Role.tool, tool_result=ToolResult(name="small", content={"ok": True})
    )
    assert "press e to expand" not in presenter.format_step(step, 0)


def _demo_results(tmp_path: Path) -> Path:
    result = SuiteResult(
        suite=SuiteMetadata(id="demo", name="Demo Suite"),
        task_results=[
            TaskResult(
                task_id="all_pass",
                trials=[_trial_result(passed=True)],
                pass_rate=1.0,
                avg_score=1.0,
            ),
            TaskResult(
                task_id="flaky",
                trials=[_trial_result(passed=True), _trial_result(passed=False, index=1)],
                pass_rate=0.5,
                avg_score=0.7,
            ),
        ],
        metrics=MetricsSummary(total_tasks=2, total_trials=3, k=2),
    )
    return write_results(result, tmp_path)


async def test_app_selects_first_failure_on_mount(tmp_path: Path) -> None:
    pytest.importorskip("textual")
    from agent_eval.ui.app import EvalBrowserApp

    app = EvalBrowserApp(_demo_results(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app._current is not None
        assert not app._current.passed  # failure-first selection
        await pilot.press("q")


async def test_app_failures_filter_and_jumps(tmp_path: Path) -> None:
    pytest.importorskip("textual")
    from agent_eval.ui.app import EvalBrowserApp

    app = EvalBrowserApp(_demo_results(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert len(app._trial_nodes) == 3
        await pilot.press("f")  # failures only
        assert len(app._trial_nodes) == 1
        assert all(n.data is not None and not n.data.passed for n in app._trial_nodes)
        await pilot.press("f")  # back to all trials
        assert len(app._trial_nodes) == 3
        await pilot.press("n")  # jump wraps to the (only) failure
        assert app._current is not None and not app._current.passed
        await pilot.press("e")  # expand toggle re-renders without error
        await pilot.press("q")


def test_live_trial_label_states() -> None:
    assert "pending" in presenter.live_trial_label(0, "pending")
    assert "running" in presenter.live_trial_label(0, "running")
    done = presenter.live_trial_label(1, "done", _trial_result(passed=False, index=1))
    assert "FAIL" in done and "score 0.70" in done


def test_live_progress_counts() -> None:
    text = presenter.live_progress(3, 6, 2, 12.34)
    assert "3/6 trials" in text
    assert "[green]2[/green]" in text
    assert "[red]1[/red]" in text
    assert "12.3s" in text


async def test_live_run_app_completes_suite() -> None:
    pytest.importorskip("textual")
    import agent_eval.graders  # noqa: F401 - register graders
    from agent_eval.harness import RunConfig
    from agent_eval.suite_loader import load_suite
    from agent_eval.ui.live import LiveRunApp

    suite = load_suite(Path("examples/suites/refund_support.yaml"))
    app = LiveRunApp(suite, RunConfig(agent="echo", concurrency=2))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
    result = app.return_value
    assert isinstance(result, SuiteResult)
    assert result.metrics.total_trials == 6
    assert app._done == 6
    assert app._passed == 6


async def test_live_run_app_abort_returns_none() -> None:
    pytest.importorskip("textual")
    import agent_eval.graders  # noqa: F401 - register graders
    from agent_eval.harness import RunConfig
    from agent_eval.suite_loader import load_suite
    from agent_eval.ui.live import LiveRunApp

    suite = load_suite(Path("examples/suites/refund_support.yaml"))
    app = LiveRunApp(suite, RunConfig(agent="echo"))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("q")  # abort immediately
    assert app.return_value is None


def test_discover_runs_lists_newest_first(tmp_path: Path) -> None:
    import os

    from agent_eval.storage import discover_runs

    old = _demo_results(tmp_path / "old")
    new = _demo_results(tmp_path / "new")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "results.json").write_text("not json", encoding="utf-8")

    runs = discover_runs(tmp_path)
    assert [r.path for r in runs] == [new, old]  # junk skipped, newest first
    assert runs[0].suite_name == "Demo Suite"
    assert runs[0].total_trials == 3


def test_run_label_summarizes_run(tmp_path: Path) -> None:
    from agent_eval.storage import discover_runs

    _demo_results(tmp_path)
    label = presenter.run_label(discover_runs(tmp_path)[0])
    assert "Demo Suite" in label
    assert "of 3 trials" in label
    assert "results.json" in label


async def test_run_picker_returns_selected_path(tmp_path: Path) -> None:
    pytest.importorskip("textual")
    from agent_eval.storage import discover_runs
    from agent_eval.ui.picker import RunPickerApp

    _demo_results(tmp_path / "a")
    _demo_results(tmp_path / "b")
    runs = discover_runs(tmp_path)
    app = RunPickerApp(runs)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")  # choose the highlighted (first) run
    assert app.return_value == runs[0].path


async def test_run_picker_quit_returns_none(tmp_path: Path) -> None:
    pytest.importorskip("textual")
    from agent_eval.storage import discover_runs
    from agent_eval.ui.picker import RunPickerApp

    _demo_results(tmp_path)
    app = RunPickerApp(discover_runs(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("q")
    assert app.return_value is None


def _compare_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Baseline where 'flaky' passes both trials; current where it regressed
    (one failure, with a different tool sequence)."""
    baseline = SuiteResult(
        suite=SuiteMetadata(id="demo", name="Demo Suite"),
        task_results=[
            TaskResult(
                task_id="flaky",
                trials=[_trial_result(passed=True), _trial_result(passed=True, index=1)],
                pass_rate=1.0,
                avg_score=1.0,
            ),
            TaskResult(
                task_id="stable",
                trials=[_trial_result(passed=True)],
                pass_rate=1.0,
                avg_score=1.0,
            ),
        ],
        metrics=MetricsSummary(total_tasks=2, total_trials=3, pass_rate=1.0, k=2),
    )
    current = SuiteResult(
        suite=SuiteMetadata(id="demo", name="Demo Suite"),
        task_results=[
            TaskResult(
                task_id="flaky",
                trials=[
                    _trial_result(passed=True),
                    _trial_result(passed=False, index=1, tool="charge_card"),
                ],
                pass_rate=0.5,
                avg_score=0.7,
            ),
            TaskResult(
                task_id="stable",
                trials=[_trial_result(passed=True)],
                pass_rate=1.0,
                avg_score=1.0,
            ),
        ],
        metrics=MetricsSummary(total_tasks=2, total_trials=3, pass_rate=2 / 3, k=2),
    )
    return (
        write_results(baseline, tmp_path / "baseline"),
        write_results(current, tmp_path / "current"),
    )


def test_compare_task_label_marks_direction() -> None:
    from agent_eval.compare import TaskDelta

    down = TaskDelta(task_id="t", status="both", baseline=1.0, current=0.5, regressed=True)
    up = TaskDelta(task_id="t", status="both", baseline=0.5, current=1.0, regressed=False)
    new = TaskDelta(task_id="t", status="new", baseline=0.0, current=1.0, regressed=False)
    assert "▼ -50.0%" in presenter.compare_task_label(down)
    assert "▲ +50.0%" in presenter.compare_task_label(up)
    assert "new" in presenter.compare_task_label(new)


def test_tool_sequence_diff() -> None:
    same = presenter.tool_sequence_diff(["a", "b"], ["a", "b"])
    assert "identical" in same
    diff = presenter.tool_sequence_diff(["a", "b"], ["a", "c"])
    assert "[red]- b[/red]" in diff
    assert "[green]+ c[/green]" in diff


def test_compare_summary_flags_regression(tmp_path: Path) -> None:
    from agent_eval.compare import compare_results
    from agent_eval.storage import load_results

    base_path, cur_path = _compare_pair(tmp_path)
    baseline, current = load_results(base_path), load_results(cur_path)
    text = presenter.compare_summary(compare_results(baseline, current), baseline, current)
    assert "REGRESSED" in text
    assert "pass_rate" in text
    assert "latency" in text


async def test_compare_app_selects_first_regression(tmp_path: Path) -> None:
    pytest.importorskip("textual")
    from agent_eval.ui.compare_view import CompareApp

    base_path, cur_path = _compare_pair(tmp_path)
    app = CompareApp(base_path, cur_path)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        tree = app.query_one("#nav")
        assert tree.cursor_node is not None and tree.cursor_node.data is not None
        assert tree.cursor_node.data.task_id == "flaky"  # regression-first
        assert tree.cursor_node.data.regressed
        detail = str(app.query_one("#detail").content)
        assert "charge_card" in detail  # tool-sequence diff rendered
        await pilot.press("n")  # wraps back to the only regression
        assert tree.cursor_node.data.task_id == "flaky"
        await pilot.press("q")
