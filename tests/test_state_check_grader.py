import pytest

from agent_eval.graders.state_check import StateCheckGrader
from agent_eval.schemas import GraderConfig, Outcome, Trial


def _trial(state: dict) -> Trial:
    return Trial(task_id="t", index=0, outcome=Outcome(state=state))


def _grader(expect: dict, **opts: object) -> StateCheckGrader:
    return StateCheckGrader(GraderConfig(type="state_check", expect=expect, **opts))


async def test_all_dot_paths_match() -> None:
    trial = _trial({"refund": {"status": "processed", "order_id": "A100"}})
    grader = _grader({"refund.status": "processed", "refund.order_id": "A100"})
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert result.passed
    assert result.score == 1.0


async def test_partial_credit_on_mismatch() -> None:
    trial = _trial({"refund": {"status": "processed"}, "ticket": {"status": "open"}})
    grader = _grader({"refund.status": "processed", "ticket.status": "resolved"})
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert not result.passed
    assert result.score == pytest.approx(0.5)


async def test_missing_path_reported() -> None:
    trial = _trial({"refund": {"status": "processed"}})
    grader = _grader({"ticket.status": "resolved"})
    result = await grader.grade(None, trial)  # type: ignore[arg-type]
    assert not result.passed
    assert "missing" in result.reason
