"""Trace failure taxonomy: classify *why* a failing trial failed.

A pass/fail flag says a trial failed; it does not say how. This module inspects
a failing trial's transcript and grader results and assigns a single best-fit
failure category, identifies the first step where the run went wrong, and lets
callers aggregate failure modes across a whole run.

Categories (checked in priority order, first match wins):

- ``timeout``           -- the trial timed out.
- ``recovery_failure``  -- a tool/step errored and the agent never recovered.
- ``looping``           -- the same tool call repeated past a threshold.
- ``policy_violation``  -- a forbidden tool was used or a hard-fail grader fired.
- ``wrong_tool``        -- a required tool was never called.
- ``wrong_args``        -- a tool was called with invalid/mismatched arguments.
- ``state_mismatch``    -- the final observable state did not match expectations.
- ``other``             -- failed for a reason none of the above explain.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_eval.schemas import Task, TaskResult, TranscriptStep, Trial, TrialResult

# How many repeats of one identical tool call count as a loop.
LOOP_THRESHOLD = 3


class FailureCategory(StrEnum):
    """The kind of failure a trial exhibited."""

    timeout = "timeout"
    recovery_failure = "recovery_failure"
    looping = "looping"
    policy_violation = "policy_violation"
    wrong_tool = "wrong_tool"
    wrong_args = "wrong_args"
    state_mismatch = "state_mismatch"
    other = "other"


@dataclass(frozen=True)
class FailureClassification:
    """The classification of a single failing trial."""

    category: FailureCategory
    reason: str
    first_bad_step: int | None  # transcript step index, or None if not localizable
    signals: list[str] = field(default_factory=list)


def _failed_grader(trial_result: TrialResult, grader_type: str) -> bool:
    return any(
        gr.grader_type == grader_type and gr.enabled and not gr.passed
        for gr in trial_result.grader_results
    )


def _grader_detail(trial_result: TrialResult, grader_type: str, key: str) -> Any:
    for gr in trial_result.grader_results:
        if gr.grader_type == grader_type and key in gr.details:
            return gr.details[key]
    return None


def _first_error_step(steps: list[TranscriptStep]) -> int | None:
    for i, step in enumerate(steps):
        if step.error:
            return i
    return None


def _first_repeated_call_step(trial: Trial) -> tuple[int | None, str | None]:
    """Index of the call that tips a tool call over ``LOOP_THRESHOLD`` repeats."""
    seen: Counter[tuple[str, str]] = Counter()
    for i, step in enumerate(trial.transcript.steps):
        if step.tool_call is None:
            continue
        key = (step.tool_call.name, repr(sorted(step.tool_call.arguments.items())))
        seen[key] += 1
        if seen[key] >= LOOP_THRESHOLD:
            return i, step.tool_call.name
    return None, None


def _first_tool_step(trial: Trial, name: str | None = None) -> int | None:
    for i, step in enumerate(trial.transcript.steps):
        if step.tool_call is not None and (name is None or step.tool_call.name == name):
            return i
    return None


def classify_failure(task: Task, trial_result: TrialResult) -> FailureClassification | None:
    """Classify a failing trial. Returns ``None`` if the trial passed."""
    if trial_result.passed:
        return None
    trial = trial_result.trial
    steps = trial.transcript.steps

    # 1. Timeout -- explicit error text wins over everything else.
    err = (trial.error or "").lower()
    if "timeout" in err or "timed out" in err:
        return FailureClassification(
            FailureCategory.timeout, trial.error or "Trial timed out.", _first_error_step(steps)
        )

    # 2. Recovery failure -- a step errored and the trial still failed.
    error_step = _first_error_step(steps)
    if error_step is not None:
        msg = steps[error_step].error or "tool error"
        return FailureClassification(
            FailureCategory.recovery_failure,
            f"Step {error_step} errored ('{msg}') and the agent did not recover.",
            error_step,
            signals=[f"error@{error_step}"],
        )

    # 3. Looping -- the same tool call repeated past the threshold.
    loop_step, loop_tool = _first_repeated_call_step(trial)
    if loop_step is not None:
        return FailureClassification(
            FailureCategory.looping,
            f"Tool '{loop_tool}' was called {LOOP_THRESHOLD}+ times with identical arguments.",
            loop_step,
            signals=[f"loop:{loop_tool}"],
        )

    # 4. Policy violation -- forbidden tool used or any hard-fail grader fired.
    forbidden = _grader_detail(trial_result, "tool_calls", "forbidden_hits") or []
    hard_fail = any(gr.hard_fail and not gr.passed for gr in trial_result.grader_results)
    if forbidden:
        name = str(forbidden[0])
        return FailureClassification(
            FailureCategory.policy_violation,
            f"Forbidden tool '{name}' was called.",
            _first_tool_step(trial, name),
            signals=[f"forbidden:{name}"],
        )
    if hard_fail:
        return FailureClassification(
            FailureCategory.policy_violation,
            "A hard-fail grader rejected the trial.",
            _first_tool_step(trial),
        )

    # 5. Wrong tool -- a required tool was never called.
    missing = _grader_detail(trial_result, "tool_calls", "missing") or []
    if missing:
        return FailureClassification(
            FailureCategory.wrong_tool,
            f"Required tool(s) never called: {', '.join(str(m) for m in missing)}.",
            None,
            signals=[f"missing:{m}" for m in missing],
        )

    # 6. Wrong args -- argument schema validation failed.
    if _failed_grader(trial_result, "argument_schema"):
        return FailureClassification(
            FailureCategory.wrong_args,
            "A tool was called with invalid arguments.",
            _first_tool_step(trial),
            signals=["argument_schema"],
        )

    # 7. State mismatch -- final observable state did not match expectations.
    if _failed_grader(trial_result, "state_check"):
        failures = _grader_detail(trial_result, "state_check", "failures") or []
        return FailureClassification(
            FailureCategory.state_mismatch,
            "Final state did not match expectations: " + "; ".join(str(f) for f in failures),
            None,
            signals=[str(f) for f in failures],
        )

    # 8. Fallback.
    failed_graders = [
        gr.grader_type for gr in trial_result.grader_results if gr.enabled and not gr.passed
    ]
    return FailureClassification(
        FailureCategory.other,
        "Failed graders: " + (", ".join(failed_graders) or "none recorded") + ".",
        None,
        signals=failed_graders,
    )


@dataclass
class FailureAggregate:
    """Failure modes aggregated across many trials/tasks."""

    counts: Counter[FailureCategory] = field(default_factory=Counter)
    by_task: dict[str, list[FailureCategory]] = field(default_factory=dict)
    total_failures: int = 0

    def most_common(self) -> list[tuple[FailureCategory, int]]:
        return self.counts.most_common()

    def summary(self) -> dict[str, int]:
        """Category -> count, as plain strings (JSON-friendly)."""
        return {str(cat): n for cat, n in self.counts.most_common()}


def classify_task(task: Task, task_result: TaskResult) -> list[FailureClassification]:
    """Classify every failing trial in a task result."""
    out = []
    for tr in task_result.trials:
        c = classify_failure(task, tr)
        if c is not None:
            out.append(c)
    return out


def aggregate_failures(tasks: dict[str, Task], task_results: list[TaskResult]) -> FailureAggregate:
    """Aggregate failure categories across a run.

    ``tasks`` maps task id -> ``Task`` (for tasks present in the suite); task
    results whose id is absent fall back to an empty placeholder task.
    """
    agg = FailureAggregate()
    for tr in task_results:
        task = tasks.get(tr.task_id) or Task(id=tr.task_id)
        classifications = classify_task(task, tr)
        if classifications:
            agg.by_task[tr.task_id] = [c.category for c in classifications]
        for c in classifications:
            agg.counts[c.category] += 1
            agg.total_failures += 1
    return agg
