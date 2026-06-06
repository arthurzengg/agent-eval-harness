"""Combine grader results into a per-trial score and pass/fail decision.

Two modes are supported:

- ``weighted``: the trial score is the weighted average of enabled graders'
  scores; the trial passes when the score meets the threshold AND no hard-fail
  grader failed.
- ``binary``: the trial passes only when every enabled grader passes; the score
  is the fraction of enabled graders that passed.

Disabled graders (e.g. an ``llm_rubric`` with ``enabled: false``) are excluded
from both the score and the pass decision. Partial credit is preserved because
each grader contributes its real-valued score.
"""

from __future__ import annotations

from agent_eval.schemas import GraderResult, Scoring, ScoringMode


def score_trial(results: list[GraderResult], scoring: Scoring) -> tuple[float, bool]:
    """Return ``(score, passed)`` for one trial's grader results."""
    enabled = [r for r in results if r.enabled]
    hard_failed = any(r.hard_fail and not r.passed for r in enabled)

    if not enabled:
        return 1.0, not hard_failed

    if scoring.mode == ScoringMode.binary:
        passed_count = sum(1 for r in enabled if r.passed)
        score = passed_count / len(enabled)
        passed = passed_count == len(enabled) and not hard_failed
        return score, passed

    total_weight = sum(r.weight for r in enabled)
    if total_weight <= 0:
        score = sum(r.score for r in enabled) / len(enabled)
    else:
        score = sum(r.score * r.weight for r in enabled) / total_weight
    passed = score >= scoring.pass_threshold and not hard_failed
    return score, passed
