"""Tests for the enhanced HTML report."""

from __future__ import annotations

from agent_eval.reporters.html_reporter import (
    _per_grader_detail,
    _task_tokens,
    render_html,
)
from agent_eval.schemas import (
    GraderResult,
    MetricsSummary,
    SuiteMetadata,
    SuiteResult,
    TaskResult,
    TokenUsage,
    Transcript,
    TranscriptStep,
    Trial,
    TrialResult,
)


def _trial_result(task_id: str, passed: bool, in_tok: int, out_tok: int) -> TrialResult:
    step = TranscriptStep(
        role="assistant",
        content="hi",
        token_usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok),
    )
    return TrialResult(
        trial=Trial(task_id=task_id, index=0, transcript=Transcript(steps=[step])),
        grader_results=[GraderResult(grader_type="exact_match", passed=passed, score=1.0)],
        score=1.0 if passed else 0.0,
        passed=passed,
    )


def _result() -> SuiteResult:
    t1 = TaskResult(task_id="t1", trials=[_trial_result("t1", True, 100, 50)], k=1, pass_rate=1.0)
    t2 = TaskResult(task_id="t2", trials=[_trial_result("t2", False, 200, 20)], k=1, pass_rate=0.0)
    return SuiteResult(
        suite=SuiteMetadata(id="s", name="S"),
        task_results=[t1, t2],
        metrics=MetricsSummary(total_tasks=2, total_trials=2, total_tokens=370),
    )


def test_per_grader_detail_counts() -> None:
    detail = _per_grader_detail(_result())
    assert detail == [{"type": "exact_match", "passed": 1, "total": 2, "rate": 0.5}]


def test_task_tokens_sum() -> None:
    assert _task_tokens(_result()) == {"t1": 150, "t2": 220}


def test_render_includes_interactive_controls() -> None:
    html = render_html(_result())
    assert 'id="failOnly"' in html
    assert "Expand all trials" in html
    assert "function applyFilter" in html
    assert "copyId" in html


def test_render_marks_failing_trials_for_filter() -> None:
    html = render_html(_result())
    assert 'data-pass="0"' in html  # the failing trial
    assert 'data-pass="1"' in html  # the passing trial


def test_render_shows_per_grader_aggregation_and_tokens() -> None:
    html = render_html(_result())
    assert "Per-grader aggregation" in html
    assert "1 / 2" in html  # passed / total for exact_match
    assert "220" in html  # t2 token total appears in the tasks table
