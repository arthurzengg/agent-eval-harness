"""Grader: exact string match against the trial's final output."""

from __future__ import annotations

from agent_eval.graders.base import BaseGrader
from agent_eval.schemas import GraderResult, Task, Trial


class ExactMatchGrader(BaseGrader):
    """Pass when ``final_output`` equals the expected string.

    Options:
        expected (str): the required output.
        strip (bool): trim surrounding whitespace before comparing (default True).
        case_sensitive (bool): default True.
    """

    type = "exact_match"

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        expected = self.options.get("expected")
        if expected is None:
            return self.result(score=0.0, passed=False, reason="No 'expected' value configured.")

        actual = trial.final_output
        expected_s = str(expected)
        if self.options.get("strip", True):
            actual, expected_s = actual.strip(), expected_s.strip()
        if not self.options.get("case_sensitive", True):
            actual, expected_s = actual.lower(), expected_s.lower()

        passed = actual == expected_s
        reason = "Output matched." if passed else "Output did not match expected string."
        return self.result(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=reason,
            details={"expected": expected_s, "actual": actual},
        )
