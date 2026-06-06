"""Grader: required / forbidden tool-call checks.

Tool-call *sequence* matching is supported but never required: by default
matching is unordered, and outcome/state graders can carry a task on their own.
This grader supports required tools, forbidden tools, optional ordered matching,
optional parameter-subset matching, partial credit, and detailed reasons.
"""

from __future__ import annotations

from typing import Any

from agent_eval.graders.base import BaseGrader
from agent_eval.schemas import GraderResult, Task, ToolCall, Trial


class ToolCallsGrader(BaseGrader):
    """Check the transcript's tool calls against required/forbidden specs.

    Options:
        required (list[{tool, params?}]): tools that must be called.
        forbidden (list[{tool, params?}]): tools that must not be called.
        ordered (bool): required tools must appear in order (default False).
        match_params (bool): require params subset-match (default True).
        allow_partial (bool): if True, forbidden hits do not hard-fail (default False).
    """

    type = "tool_calls"

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        required = self.options.get("required", []) or []
        forbidden = self.options.get("forbidden", []) or []
        ordered = bool(self.options.get("ordered", False))
        match_params = bool(self.options.get("match_params", True))
        allow_partial = bool(self.options.get("allow_partial", False))

        calls = trial.transcript.tool_calls()
        reasons: list[str] = []

        satisfied, missing = self._check_required(required, calls, match_params)
        forbidden_hits = self._check_forbidden(forbidden, calls, match_params)

        for spec in missing:
            reasons.append(f"Missing required tool: {spec.get('tool')}")
        for name in forbidden_hits:
            reasons.append(f"Forbidden tool called: {name}")

        order_ok = True
        if ordered and not missing:
            order_ok = self._check_order(required, calls, match_params)
            if not order_ok:
                reasons.append("Required tools were not called in the expected order.")

        total = len(required)
        score = (len(satisfied) / total) if total else 1.0
        if ordered and not order_ok:
            score = min(score, 0.5)

        has_forbidden = bool(forbidden_hits)
        hard_fail = has_forbidden and not allow_partial
        if has_forbidden and not allow_partial:
            score = 0.0
        elif has_forbidden:
            score = min(score, 0.5)

        passed = not missing and not has_forbidden and order_ok
        reason = "All tool-call checks passed." if passed else "; ".join(reasons)
        return self.result(
            score=score,
            passed=passed,
            hard_fail=hard_fail,
            reason=reason,
            details={
                "required_total": total,
                "required_satisfied": len(satisfied),
                "missing": [s.get("tool") for s in missing],
                "forbidden_hits": forbidden_hits,
                "ordered": ordered,
            },
        )

    def _check_required(
        self, required: list[dict[str, Any]], calls: list[ToolCall], match_params: bool
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        satisfied, missing = [], []
        for spec in required:
            if any(_matches(spec, call, match_params) for call in calls):
                satisfied.append(spec)
            else:
                missing.append(spec)
        return satisfied, missing

    def _check_forbidden(
        self, forbidden: list[dict[str, Any]], calls: list[ToolCall], match_params: bool
    ) -> list[str]:
        hits: list[str] = []
        for spec in forbidden:
            if any(_matches(spec, call, match_params) for call in calls):
                hits.append(str(spec.get("tool")))
        return hits

    def _check_order(
        self, required: list[dict[str, Any]], calls: list[ToolCall], match_params: bool
    ) -> bool:
        idx = 0
        for call in calls:
            if idx < len(required) and _matches(required[idx], call, match_params):
                idx += 1
        return idx == len(required)


def _matches(spec: dict[str, Any], call: ToolCall, match_params: bool) -> bool:
    if spec.get("tool") != call.name:
        return False
    if not match_params:
        return True
    params = spec.get("params") or {}
    return all(call.arguments.get(k) == v for k, v in params.items())
