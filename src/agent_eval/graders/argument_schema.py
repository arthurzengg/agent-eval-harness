"""Grader: validate tool-call arguments against a JSON Schema."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from agent_eval.graders.base import BaseGrader
from agent_eval.schemas import GraderResult, Task, Trial


class ArgumentSchemaGrader(BaseGrader):
    """Validate the arguments of calls to a given tool with JSON Schema.

    Options:
        tool (str): the tool whose calls are validated.
        schema (dict): a JSON Schema applied to each matching call's arguments.
        require_call (bool): fail if the tool was never called (default True).
    """

    type = "argument_schema"

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        tool = self.options.get("tool")
        schema = self.options.get("schema")
        if not tool or not isinstance(schema, dict):
            return self.result(
                score=0.0, passed=False, reason="Configure both 'tool' and 'schema'."
            )

        validator = Draft202012Validator(schema)
        calls = [c for c in trial.transcript.tool_calls() if c.name == tool]
        if not calls:
            require_call = self.options.get("require_call", True)
            return self.result(
                score=0.0 if require_call else 1.0,
                passed=not require_call,
                reason=f"Tool '{tool}' was never called.",
            )

        valid = 0
        errors: list[str] = []
        for i, call in enumerate(calls):
            call_errors = _validate(validator, call.arguments)
            if call_errors:
                errors.append(f"call[{i}]: {'; '.join(call_errors)}")
            else:
                valid += 1

        score = valid / len(calls)
        passed = valid == len(calls)
        reason = "All tool arguments valid." if passed else " | ".join(errors)
        return self.result(
            score=score,
            passed=passed,
            reason=reason,
            details={"valid": valid, "total": len(calls), "errors": errors},
        )


def _validate(validator: Draft202012Validator, args: dict[str, Any]) -> list[str]:
    return [e.message for e in validator.iter_errors(args)]
