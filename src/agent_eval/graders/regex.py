"""Grader: regex pattern matching against the trial's final output."""

from __future__ import annotations

import re

from agent_eval.graders.base import BaseGrader
from agent_eval.schemas import GraderResult, Task, Trial


class RegexGrader(BaseGrader):
    """Check one or more regex patterns against ``final_output``.

    Options:
        patterns (list[str] | str): patterns that must match.
        mode ("all" | "any"): require all patterns or at least one (default all).
        flags (list[str]): subset of {"IGNORECASE", "MULTILINE", "DOTALL"}.
        partial_credit (bool): score as fraction matched (default True).
    """

    type = "regex"

    def validate_config(self) -> None:
        patterns = self.options.get("patterns")
        if not patterns:
            raise ValueError("requires one or more 'patterns'.")
        candidates = [patterns] if isinstance(patterns, str) else list(patterns)
        for p in candidates:
            try:
                re.compile(p)
            except re.error as exc:
                raise ValueError(f"invalid regex {p!r}: {exc}") from exc

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        patterns = self.options.get("patterns", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if not patterns:
            return self.result(score=0.0, passed=False, reason="No 'patterns' configured.")

        flags = _resolve_flags(self.options.get("flags", []))
        text = trial.final_output
        matched = [bool(re.search(p, text, flags)) for p in patterns]
        num_matched = sum(matched)

        mode = self.options.get("mode", "all")
        passed = (num_matched == len(patterns)) if mode == "all" else (num_matched > 0)

        if self.options.get("partial_credit", True):
            score = num_matched / len(patterns)
        else:
            score = 1.0 if passed else 0.0

        missed = [p for p, m in zip(patterns, matched, strict=True) if not m]
        reason = "All patterns matched." if passed else f"Unmatched patterns: {missed}"
        return self.result(
            score=score,
            passed=passed,
            reason=reason,
            details={"matched": num_matched, "total": len(patterns), "missed": missed},
        )


def _resolve_flags(names: list[str]) -> re.RegexFlag:
    flags = re.RegexFlag(0)
    for name in names:
        flags |= getattr(re, name.upper(), re.RegexFlag(0))
    return flags
