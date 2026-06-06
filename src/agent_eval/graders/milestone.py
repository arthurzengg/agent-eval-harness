"""Grader: dynamic milestone scoring over a trial's trajectory.

Final-state graders only check where the agent ended up. Many tasks also care
about *how* it got there: did it reach the right intermediate states, in the
right order? This grader scores progress through a list of milestones, each
satisfied by an event in the transcript (a tool call, a phrase in the output) or
by the final observable state, and can enforce ordering between them.

Ordering is expressed two ways:

- ``ordered: true`` requires every milestone to be reached in listed order.
- a milestone's ``after: [other_name]`` requires it to be reached no earlier
  than the named milestone(s) -- e.g. ``refund_processed`` *after*
  ``identity_verified`` encodes "identity verified before refund processed".

Each milestone may specify:

- ``tool``: a tool name that must be called (optional ``params`` subset match);
- ``contains``: a substring that must appear in an assistant message or the
  final output;
- ``state``: a mapping of dot-path -> expected value on the final outcome state.
"""

from __future__ import annotations

from typing import Any

from agent_eval.graders.base import BaseGrader, get_dot_path
from agent_eval.schemas import GraderResult, Task, Trial

# Sentinel index for milestones satisfied by the final state (they have no
# transcript position, so they sort after every in-transcript event).
_END = 10**9


class MilestoneGrader(BaseGrader):
    """Score ordered progress through intermediate milestones."""

    type = "milestone"

    def validate_config(self) -> None:
        milestones = self.options.get("milestones")
        if not milestones or not isinstance(milestones, list):
            raise ValueError("requires a non-empty 'milestones' list.")
        names = set()
        for i, m in enumerate(milestones):
            if not isinstance(m, dict) or "name" not in m:
                raise ValueError(f"milestone[{i}] must be a mapping with a 'name'.")
            if not (m.get("tool") or m.get("contains") or m.get("state")):
                raise ValueError(
                    f"milestone '{m.get('name')}' needs one of 'tool', 'contains', or 'state'."
                )
            names.add(m["name"])
        for m in milestones:
            for dep in m.get("after", []) or []:
                if dep not in names:
                    raise ValueError(f"milestone '{m['name']}' refers to unknown 'after': {dep}.")

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        milestones: list[dict[str, Any]] = self.options.get("milestones", []) or []
        if not milestones:
            return self.result(score=1.0, passed=True, reason="No milestones configured.")
        ordered = bool(self.options.get("ordered", False))

        reached_at = {m["name"]: self._reached_at(m, trial) for m in milestones}
        reached = [m["name"] for m in milestones if reached_at[m["name"]] is not None]
        reasons: list[str] = []
        for m in milestones:
            if reached_at[m["name"]] is None:
                reasons.append(f"Milestone not reached: {m['name']}")

        order_ok = self._check_order(milestones, reached_at, ordered, reasons)

        total = len(milestones)
        score = len(reached) / total
        if not order_ok:
            score = min(score, 0.5)
        passed = len(reached) == total and order_ok
        reason = "All milestones reached in order." if passed else "; ".join(reasons)
        return self.result(
            score=score,
            passed=passed,
            reason=reason,
            details={
                "reached": reached,
                "total": total,
                "order_ok": order_ok,
                "reached_at": {k: v for k, v in reached_at.items() if v is not None},
            },
        )

    def _check_order(
        self,
        milestones: list[dict[str, Any]],
        reached_at: dict[str, int | None],
        ordered: bool,
        reasons: list[str],
    ) -> bool:
        order_ok = True
        if ordered:
            last = -1
            for m in milestones:
                at = reached_at[m["name"]]
                if at is None:
                    continue
                if at < last:
                    order_ok = False
                    reasons.append(f"Milestone out of order: {m['name']}")
                last = max(last, at)
        for m in milestones:
            at = reached_at[m["name"]]
            for dep in m.get("after", []) or []:
                dep_at = reached_at.get(dep)
                if at is None:
                    continue
                if dep_at is None or dep_at > at:
                    order_ok = False
                    reasons.append(f"Milestone '{m['name']}' must come after '{dep}'")
        return order_ok

    def _reached_at(self, milestone: dict[str, Any], trial: Trial) -> int | None:
        """First transcript index at which ``milestone`` is satisfied, else None."""
        if "tool" in milestone:
            return self._tool_index(milestone, trial)
        if "contains" in milestone:
            return self._contains_index(str(milestone["contains"]), trial)
        if "state" in milestone:
            return _END if self._state_satisfied(milestone["state"], trial) else None
        return None

    def _tool_index(self, milestone: dict[str, Any], trial: Trial) -> int | None:
        tool = milestone["tool"]
        params = milestone.get("params") or {}
        for i, step in enumerate(trial.transcript.steps):
            call = step.tool_call
            if call is None or call.name != tool:
                continue
            if all(call.arguments.get(k) == v for k, v in params.items()):
                return i
        return None

    def _contains_index(self, needle: str, trial: Trial) -> int | None:
        for i, step in enumerate(trial.transcript.steps):
            if isinstance(step.content, str) and needle in step.content:
                return i
        if needle in trial.final_output:
            return _END
        return None

    def _state_satisfied(self, expect: dict[str, Any], trial: Trial) -> bool:
        for path, expected in expect.items():
            found, actual = get_dot_path(trial.outcome.state, path)
            if not found or actual != expected:
                return False
        return True
