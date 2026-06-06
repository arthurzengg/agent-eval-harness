"""Tests for the trace failure taxonomy."""

from __future__ import annotations

from agent_eval.schemas import (
    GraderResult,
    Role,
    Task,
    TaskResult,
    ToolCall,
    Transcript,
    TranscriptStep,
    Trial,
    TrialResult,
)
from agent_eval.taxonomy import (
    FailureCategory,
    aggregate_failures,
    classify_failure,
)


def _step(*, tool: str | None = None, args: dict | None = None, error: str | None = None):
    call = ToolCall(name=tool, arguments=args or {}) if tool else None
    return TranscriptStep(role=Role.assistant, tool_call=call, error=error)


def _trial_result(
    *,
    passed: bool,
    steps: list[TranscriptStep] | None = None,
    error: str | None = None,
    graders: list[GraderResult] | None = None,
) -> TrialResult:
    trial = Trial(
        task_id="t1",
        index=0,
        transcript=Transcript(steps=steps or []),
        error=error,
    )
    return TrialResult(trial=trial, grader_results=graders or [], passed=passed)


def test_passing_trial_is_not_classified() -> None:
    assert classify_failure(Task(id="t1"), _trial_result(passed=True)) is None


def test_timeout_classification() -> None:
    c = classify_failure(
        Task(id="t1"), _trial_result(passed=False, error="Trial timed out after 60s")
    )
    assert c is not None
    assert c.category == FailureCategory.timeout


def test_recovery_failure_classification() -> None:
    steps = [_step(tool="lookup"), _step(error="connection refused")]
    c = classify_failure(Task(id="t1"), _trial_result(passed=False, steps=steps))
    assert c is not None
    assert c.category == FailureCategory.recovery_failure
    assert c.first_bad_step == 1


def test_looping_classification() -> None:
    steps = [_step(tool="search", args={"q": "x"}) for _ in range(3)]
    c = classify_failure(Task(id="t1"), _trial_result(passed=False, steps=steps))
    assert c is not None
    assert c.category == FailureCategory.looping
    assert c.first_bad_step == 2  # third identical call tips the threshold


def test_policy_violation_from_forbidden_tool() -> None:
    steps = [_step(tool="delete_account", args={})]
    graders = [
        GraderResult(
            grader_type="tool_calls",
            passed=False,
            hard_fail=True,
            details={"forbidden_hits": ["delete_account"], "missing": []},
        )
    ]
    c = classify_failure(Task(id="t1"), _trial_result(passed=False, steps=steps, graders=graders))
    assert c is not None
    assert c.category == FailureCategory.policy_violation
    assert c.first_bad_step == 0


def test_wrong_tool_from_missing_required() -> None:
    graders = [
        GraderResult(
            grader_type="tool_calls",
            passed=False,
            details={"forbidden_hits": [], "missing": ["verify_identity"]},
        )
    ]
    c = classify_failure(Task(id="t1"), _trial_result(passed=False, graders=graders))
    assert c is not None
    assert c.category == FailureCategory.wrong_tool
    assert "verify_identity" in c.reason


def test_wrong_args_from_argument_schema() -> None:
    steps = [_step(tool="process_refund", args={"amount": "lots"})]
    graders = [GraderResult(grader_type="argument_schema", passed=False)]
    c = classify_failure(Task(id="t1"), _trial_result(passed=False, steps=steps, graders=graders))
    assert c is not None
    assert c.category == FailureCategory.wrong_args
    assert c.first_bad_step == 0


def test_state_mismatch_classification() -> None:
    graders = [
        GraderResult(
            grader_type="state_check",
            passed=False,
            details={"failures": ["refund.status = 'denied', expected 'processed'"]},
        )
    ]
    c = classify_failure(Task(id="t1"), _trial_result(passed=False, graders=graders))
    assert c is not None
    assert c.category == FailureCategory.state_mismatch
    assert "refund.status" in c.reason


def test_other_fallback() -> None:
    graders = [GraderResult(grader_type="regex", passed=False)]
    c = classify_failure(Task(id="t1"), _trial_result(passed=False, graders=graders))
    assert c is not None
    assert c.category == FailureCategory.other
    assert "regex" in c.reason


def test_priority_timeout_beats_state() -> None:
    graders = [GraderResult(grader_type="state_check", passed=False, details={"failures": ["x"]})]
    c = classify_failure(
        Task(id="t1"), _trial_result(passed=False, error="request timeout", graders=graders)
    )
    assert c is not None
    assert c.category == FailureCategory.timeout


def test_aggregate_failures_across_tasks() -> None:
    t1 = TaskResult(
        task_id="t1",
        k=2,
        trials=[
            _trial_result(passed=False, error="timeout"),
            _trial_result(passed=True),
        ],
    )
    t2 = TaskResult(
        task_id="t2",
        k=1,
        trials=[
            _trial_result(
                passed=False,
                graders=[
                    GraderResult(grader_type="state_check", passed=False, details={"failures": []})
                ],
            )
        ],
    )
    agg = aggregate_failures({"t1": Task(id="t1")}, [t1, t2])
    assert agg.total_failures == 2
    assert agg.counts[FailureCategory.timeout] == 1
    assert agg.counts[FailureCategory.state_mismatch] == 1
    assert "t1" in agg.by_task and "t2" in agg.by_task
    assert agg.summary()["timeout"] == 1
