"""Tests for scoring modes, the console reporter, and the mock LLM judge."""

from __future__ import annotations

import asyncio

from rich.console import Console

from agent_eval.graders.llm_rubric import LLMRubricGrader, MockProvider, resolve_provider
from agent_eval.reporters.console_reporter import ConsoleReporter
from agent_eval.schemas import (
    GraderConfig,
    GraderResult,
    MetricsSummary,
    Scoring,
    ScoringMode,
    SuiteMetadata,
    SuiteResult,
    Task,
    TaskResult,
    Trial,
    TrialResult,
)
from agent_eval.scoring import score_trial


def _gr(score: float, passed: bool, weight: float = 1.0, **kw: object) -> GraderResult:
    return GraderResult(grader_type="g", score=score, passed=passed, weight=weight, **kw)  # type: ignore[arg-type]


def test_weighted_score_and_pass() -> None:
    results = [_gr(1.0, True, 3.0), _gr(0.0, False, 1.0)]
    score, passed = score_trial(results, Scoring(mode=ScoringMode.weighted, pass_threshold=0.7))
    assert round(score, 3) == 0.75
    assert passed is True


def test_weighted_below_threshold_fails() -> None:
    score, passed = score_trial([_gr(0.5, False)], Scoring(pass_threshold=0.8))
    assert passed is False


def test_hard_fail_sinks_pass() -> None:
    results = [_gr(1.0, True), _gr(0.0, False, hard_fail=True)]
    score, passed = score_trial(results, Scoring(pass_threshold=0.1))
    assert passed is False


def test_binary_requires_all_pass() -> None:
    score, passed = score_trial([_gr(1.0, True), _gr(0.0, False)], Scoring(mode=ScoringMode.binary))
    assert score == 0.5
    assert passed is False


def test_no_enabled_graders_passes() -> None:
    score, passed = score_trial([_gr(0.0, False, enabled=False)], Scoring())
    assert score == 1.0
    assert passed is True


def test_zero_total_weight_falls_back_to_mean() -> None:
    score, _ = score_trial([_gr(0.4, False, weight=0.0), _gr(0.6, True, weight=0.0)], Scoring())
    assert round(score, 3) == 0.5


def _suite_result(passed: bool) -> SuiteResult:
    trial = TrialResult(
        trial=Trial(task_id="t1", index=0),
        grader_results=[GraderResult(grader_type="exact_match", passed=passed, reason="nope")],
        passed=passed,
    )
    tr = TaskResult(task_id="t1", trials=[trial], k=1, pass_rate=1.0 if passed else 0.0)
    return SuiteResult(
        suite=SuiteMetadata(id="s", name="S"),
        task_results=[tr],
        metrics=MetricsSummary(total_tasks=1, total_trials=1, k_min=1, k_max=1),
    )


def test_console_reporter_all_passed() -> None:
    console = Console(record=True, width=100)
    ConsoleReporter(console).render(_suite_result(True))
    out = console.export_text()
    assert "All tasks passed every trial." in out


def test_console_reporter_with_failures() -> None:
    console = Console(record=True, width=100)
    ConsoleReporter(console).render(_suite_result(False))
    out = console.export_text()
    assert "Tasks with failing trials" in out
    assert "failure" in out  # the "Top failure reasons" table
    assert "nope" in out  # the grader's failure reason


def test_console_reporter_mixed_k_note() -> None:
    console = Console(record=True, width=120)
    result = _suite_result(True)
    result.metrics.k_min = 2
    result.metrics.k_max = 4
    result.metrics.consistent_k = False
    ConsoleReporter(console).render(result)
    assert "mixed" in console.export_text()


def test_mock_judge_returns_unknown() -> None:
    provider = MockProvider()
    verdict = provider.judge("sys", '{"assertions": ["a1", "a2"]}')
    assert all(r["verdict"] == "Unknown" for r in verdict["assertion_results"])


def test_resolve_provider_defaults_to_mock() -> None:
    assert isinstance(resolve_provider(), MockProvider)


def test_llm_rubric_with_mock_provider_is_non_passing() -> None:
    # The mock judge answers Unknown for every assertion, so it never passes.
    grader = LLMRubricGrader(GraderConfig(type="llm_rubric", assertions=["x"]))
    result = asyncio.run(grader.grade(Task(id="t1"), Trial(task_id="t1", index=0)))
    assert result.passed is False
    assert result.details["unknown"] == 1


def test_llm_rubric_no_assertions_is_neutral_fail() -> None:
    grader = LLMRubricGrader(GraderConfig(type="llm_rubric"))
    result = asyncio.run(grader.grade(Task(id="t1"), Trial(task_id="t1", index=0)))
    assert result.passed is False
    assert "No assertions" in result.reason


def test_llm_rubric_respects_enabled_flag() -> None:
    grader = LLMRubricGrader(GraderConfig(type="llm_rubric", enabled=False, assertions=["x"]))
    result = asyncio.run(grader.grade(Task(id="t1"), Trial(task_id="t1", index=0)))
    assert result.enabled is False
