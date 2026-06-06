"""Tests for the dynamic milestone grader."""

from __future__ import annotations

import asyncio

import pytest

from agent_eval.graders.milestone import MilestoneGrader
from agent_eval.schemas import (
    GraderConfig,
    Outcome,
    Role,
    Task,
    ToolCall,
    Transcript,
    TranscriptStep,
    Trial,
)


def _grade(options: dict, trial: Trial):
    grader = MilestoneGrader(GraderConfig(type="milestone", **options))
    return asyncio.run(grader.grade(Task(id="t1"), trial))


def _tool_step(name: str, **args) -> TranscriptStep:
    return TranscriptStep(role=Role.assistant, tool_call=ToolCall(name=name, arguments=args))


def _trial(steps: list[TranscriptStep], *, final: str = "", state: dict | None = None) -> Trial:
    return Trial(
        task_id="t1",
        index=0,
        transcript=Transcript(steps=steps),
        final_output=final,
        outcome=Outcome(state=state or {}),
    )


REFUND_FLOW = {
    "ordered": True,
    "milestones": [
        {"name": "identity_verified", "tool": "verify_identity"},
        {"name": "refund_processed", "tool": "process_refund"},
    ],
}


def test_all_milestones_in_order_passes() -> None:
    trial = _trial([_tool_step("verify_identity"), _tool_step("process_refund")])
    res = _grade(REFUND_FLOW, trial)
    assert res.passed
    assert res.score == 1.0
    assert res.details["reached"] == ["identity_verified", "refund_processed"]


def test_out_of_order_fails_even_if_all_reached() -> None:
    trial = _trial([_tool_step("process_refund"), _tool_step("verify_identity")])
    res = _grade(REFUND_FLOW, trial)
    assert not res.passed
    assert res.details["order_ok"] is False
    assert res.score <= 0.5


def test_partial_progress_scores_fraction() -> None:
    trial = _trial([_tool_step("verify_identity")])
    res = _grade(REFUND_FLOW, trial)
    assert not res.passed
    assert res.score == 0.5
    assert "refund_processed" in res.reason


def test_after_predicate_encodes_before_relation() -> None:
    options = {
        "milestones": [
            {"name": "identity_verified", "tool": "verify_identity"},
            {"name": "refund_processed", "tool": "process_refund", "after": ["identity_verified"]},
        ]
    }
    good = _trial([_tool_step("verify_identity"), _tool_step("process_refund")])
    assert _grade(options, good).passed
    bad = _trial([_tool_step("process_refund"), _tool_step("verify_identity")])
    res = _grade(options, bad)
    assert not res.passed
    assert res.details["order_ok"] is False


def test_params_subset_match() -> None:
    options = {
        "milestones": [{"name": "big_refund", "tool": "process_refund", "params": {"amount": 50}}]
    }
    assert _grade(options, _trial([_tool_step("process_refund", amount=50)])).passed
    assert not _grade(options, _trial([_tool_step("process_refund", amount=10)])).passed


def test_contains_milestone_from_final_output() -> None:
    options = {"milestones": [{"name": "apology", "contains": "sorry"}]}
    assert _grade(options, _trial([], final="I am sorry for the trouble")).passed
    assert not _grade(options, _trial([], final="no issue")).passed


def test_state_milestone() -> None:
    options = {"milestones": [{"name": "done", "state": {"refund.status": "processed"}}]}
    assert _grade(options, _trial([], state={"refund": {"status": "processed"}})).passed
    assert not _grade(options, _trial([], state={"refund": {"status": "denied"}})).passed


def test_validate_requires_milestones() -> None:
    with pytest.raises(ValueError, match="milestones"):
        MilestoneGrader(GraderConfig(type="milestone")).validate_config()


def test_validate_requires_predicate() -> None:
    grader = MilestoneGrader(GraderConfig(type="milestone", milestones=[{"name": "x"}]))
    with pytest.raises(ValueError, match="needs one of"):
        grader.validate_config()


def test_validate_unknown_after() -> None:
    grader = MilestoneGrader(
        GraderConfig(
            type="milestone",
            milestones=[{"name": "a", "tool": "x", "after": ["ghost"]}],
        )
    )
    with pytest.raises(ValueError, match="unknown 'after'"):
        grader.validate_config()


def test_validate_accepts_valid_config() -> None:
    MilestoneGrader(GraderConfig(type="milestone", **REFUND_FLOW)).validate_config()
