"""Grader: structural / budget checks on the transcript."""

from __future__ import annotations

import json

from agent_eval.graders.base import BaseGrader
from agent_eval.schemas import GraderResult, Role, Task, Trial


class TranscriptGrader(BaseGrader):
    """Check transcript budgets and required/forbidden phrases.

    Options (all optional):
        max_turns (int): maximum assistant/user turns.
        max_tool_calls (int): maximum tool calls.
        max_errors (int): maximum error steps.
        max_duration_ms (float): maximum total step duration, when recorded.
        required_phrases (list[str]): phrases that must appear.
        forbidden_phrases (list[str]): phrases that must not appear.

    Score is the fraction of configured checks that pass.
    """

    type = "transcript"

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        checks: list[tuple[bool, str]] = []
        t = trial.transcript

        self._budget(checks, "max_turns", t.turn_count(), "turns")
        self._budget(checks, "max_tool_calls", len(t.tool_calls()), "tool calls")
        self._budget(checks, "max_errors", t.error_count(), "errors")
        self._budget(checks, "max_duration_ms", t.total_duration_ms(), "duration_ms")

        text = _transcript_text(trial)
        for phrase in self.options.get("required_phrases", []) or []:
            checks.append((phrase in text, f"required phrase missing: {phrase!r}"))
        for phrase in self.options.get("forbidden_phrases", []) or []:
            checks.append((phrase not in text, f"forbidden phrase present: {phrase!r}"))

        if not checks:
            return self.result(score=1.0, passed=True, reason="No transcript checks configured.")

        passed_count = sum(1 for ok, _ in checks if ok)
        failures = [msg for ok, msg in checks if not ok]
        passed = passed_count == len(checks)
        reason = "All transcript checks passed." if passed else "; ".join(failures)
        return self.result(
            score=passed_count / len(checks),
            passed=passed,
            reason=reason,
            details={"passed": passed_count, "total": len(checks), "failures": failures},
        )

    def _budget(self, checks: list[tuple[bool, str]], key: str, actual: float, label: str) -> None:
        limit = self.options.get(key)
        if limit is not None:
            checks.append((actual <= limit, f"{label} {actual} exceeds limit {limit}"))


def _transcript_text(trial: Trial) -> str:
    parts: list[str] = [trial.final_output]
    for step in trial.transcript.steps:
        if step.role == Role.assistant and step.content is not None:
            parts.append(
                step.content if isinstance(step.content, str) else json.dumps(step.content)
            )
    return "\n".join(parts)
