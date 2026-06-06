import pytest

from agent_eval.graders.tool_calls import ToolCallsGrader
from agent_eval.schemas import (
    GraderConfig,
    Outcome,
    Role,
    ToolCall,
    Transcript,
    TranscriptStep,
    Trial,
)


def _trial(*tool_calls: tuple[str, dict]) -> Trial:
    steps = [
        TranscriptStep(role=Role.assistant, tool_call=ToolCall(name=name, arguments=args))
        for name, args in tool_calls
    ]
    return Trial(task_id="t", index=0, transcript=Transcript(steps=steps), outcome=Outcome())


def _grader(**opts: object) -> ToolCallsGrader:
    return ToolCallsGrader(GraderConfig(type="tool_calls", **opts))


async def test_required_present_passes() -> None:
    trial = _trial(("verify_identity", {}), ("process_refund", {"order_id": "A100"}))
    grader = _grader(
        required=[
            {"tool": "verify_identity"},
            {"tool": "process_refund", "params": {"order_id": "A100"}},
        ]
    )
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert result.passed
    assert result.score == 1.0


async def test_missing_required_partial_credit() -> None:
    trial = _trial(("verify_identity", {}))
    grader = _grader(required=[{"tool": "verify_identity"}, {"tool": "process_refund"}])
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert not result.passed
    assert result.score == pytest.approx(0.5)
    assert "process_refund" in result.reason


async def test_forbidden_hard_fails() -> None:
    trial = _trial(("verify_identity", {}), ("escalate_to_manager", {}))
    grader = _grader(
        required=[{"tool": "verify_identity"}], forbidden=[{"tool": "escalate_to_manager"}]
    )
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert not result.passed
    assert result.hard_fail
    assert result.score == 0.0


async def test_param_mismatch_not_matched() -> None:
    trial = _trial(("process_refund", {"order_id": "B200"}))
    grader = _grader(required=[{"tool": "process_refund", "params": {"order_id": "A100"}}])
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert not result.passed


async def test_ordered_matching() -> None:
    trial = _trial(("b", {}), ("a", {}))
    grader = _grader(required=[{"tool": "a"}, {"tool": "b"}], ordered=True)
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert not result.passed
    assert "order" in result.reason.lower()
