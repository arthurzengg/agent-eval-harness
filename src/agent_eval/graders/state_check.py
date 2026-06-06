"""Grader: dot-path expectations against the trial outcome/state."""

from __future__ import annotations

from typing import Any

from agent_eval.graders.base import BaseGrader, get_dot_path
from agent_eval.schemas import GraderResult, Task, Trial


class StateCheckGrader(BaseGrader):
    """Check outcome state using dot-path expectations.

    Options:
        expect (dict[str, Any]): mapping of dot-path -> expected value, e.g.
            ``{"refund.status": "processed", "ticket.status": "resolved"}``.
        partial_credit (bool): score as fraction satisfied (default True).

    The state checked is the trial's outcome state. The harness mirrors the
    environment's final state into the outcome, so this also covers
    environment-observed state.
    """

    type = "state_check"

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        expect: dict[str, Any] = self.options.get("expect", {}) or {}
        if not expect:
            return self.result(score=1.0, passed=True, reason="No expectations configured.")

        state = trial.outcome.state
        failures: list[str] = []
        satisfied = 0
        for path, expected in expect.items():
            found, actual = get_dot_path(state, path)
            if found and actual == expected:
                satisfied += 1
            elif not found:
                failures.append(f"{path} missing (expected {expected!r})")
            else:
                failures.append(f"{path} = {actual!r}, expected {expected!r}")

        total = len(expect)
        score = (
            satisfied / total
            if self.options.get("partial_credit", True)
            else (1.0 if satisfied == total else 0.0)
        )
        passed = satisfied == total
        reason = "All state expectations met." if passed else "; ".join(failures)
        return self.result(
            score=score,
            passed=passed,
            reason=reason,
            details={"satisfied": satisfied, "total": total, "failures": failures},
        )
