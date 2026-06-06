"""Grader: required / forbidden tool-call checks.

Tool-call *sequence* matching is supported but never required: by default
matching is unordered, and outcome/state graders can carry a task on their own.
This grader supports required tools, forbidden tools, optional ordered matching,
optional parameter-subset matching, partial credit, and detailed reasons.
"""

from __future__ import annotations

from typing import Any

from agent_eval.canonicalize import ArgSpec, args_match, parse_call
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

    Each required/forbidden entry may instead be written as an AST-style call
    expression -- ``{call: "process_refund(amount=50)"}`` -- and may carry
    semantic matching for its params via ``match`` (field -> kind, e.g.
    ``{amount: amount, when: date, order_id: id}``), an ``aliases`` table per
    field, or ``fuzzy: true`` (optionally ``fuzzy_threshold``) for whole-arg
    fuzzy text equality. See ``agent_eval.canonicalize``.
    """

    type = "tool_calls"

    def validate_config(self) -> None:
        if not self.options.get("required") and not self.options.get("forbidden"):
            raise ValueError("requires at least one of 'required' or 'forbidden'.")
        for key in ("required", "forbidden"):
            for item in self.options.get(key, []) or []:
                if not isinstance(item, dict) or not ("tool" in item or "call" in item):
                    raise ValueError(
                        f"each '{key}' entry must be a mapping with a 'tool' or 'call' key."
                    )
                if "call" in item:
                    try:
                        parse_call(str(item["call"]))
                    except ValueError as exc:
                        raise ValueError(f"invalid 'call' expression: {exc}") from exc

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
            reasons.append(f"Missing required tool: {_spec_name(spec)}")
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
                "missing": [_spec_name(s) for s in missing],
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
                hits.append(_spec_name(spec))
        return hits

    def _check_order(
        self, required: list[dict[str, Any]], calls: list[ToolCall], match_params: bool
    ) -> bool:
        idx = 0
        for call in calls:
            if idx < len(required) and _matches(required[idx], call, match_params):
                idx += 1
        return idx == len(required)


def _spec_name(spec: dict[str, Any]) -> str:
    """Display name for a spec, resolving an AST 'call' expression if present."""
    name, _ = _resolve_spec(spec)
    return str(name)


def _resolve_spec(spec: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Return the (tool_name, params) for a spec, supporting AST 'call' exprs."""
    if "call" in spec:
        parsed = parse_call(str(spec["call"]))
        return parsed.name, parsed.kwargs
    return spec.get("tool"), (spec.get("params") or {})


def _arg_specs(spec: dict[str, Any]) -> tuple[dict[str, ArgSpec], ArgSpec]:
    """Build per-field and default ``ArgSpec``s from a spec's match options."""
    aliases = spec.get("aliases") or {}
    per_field = {
        field: ArgSpec(kind=str(kind), aliases=aliases.get(field, {}))
        for field, kind in (spec.get("match") or {}).items()
    }
    if spec.get("fuzzy"):
        default = ArgSpec(kind="fuzzy", fuzzy_threshold=float(spec.get("fuzzy_threshold", 0.85)))
    else:
        default = ArgSpec()
    return per_field, default


def _matches(spec: dict[str, Any], call: ToolCall, match_params: bool) -> bool:
    tool, params = _resolve_spec(spec)
    if tool != call.name:
        return False
    if not match_params or not params:
        return True
    per_field, default = _arg_specs(spec)
    return args_match(params, call.arguments, per_field, default=default)
