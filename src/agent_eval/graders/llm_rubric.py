"""Grader: optional LLM-as-judge rubric grading.

This grader is DISABLED by default and only runs when its config sets
``enabled: true``. It uses a pluggable provider interface; the default provider
is a deterministic ``MockProvider`` that requires no network or API keys. Real
providers (OpenAI / Anthropic) are selected via environment variables and are
intentionally minimal — no paid model is hardcoded.

The judge is instructed to return ``Unknown`` for any assertion whose evidence
is insufficient, so that a missing signal is never silently counted as a pass.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from agent_eval.graders.base import BaseGrader
from agent_eval.schemas import GraderResult, Task, Trial

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. For each assertion, decide Yes, No, or "
    "Unknown. Answer Unknown whenever the transcript does not contain enough "
    "evidence to decide; never guess. Return ONLY JSON of the form: "
    '{"assertion_results": [{"assertion": str, "verdict": "Yes|No|Unknown", '
    '"reason": str}], "reason": str}.'
)


class JudgeProvider(Protocol):
    """Provider that returns a judge verdict as a JSON-compatible dict."""

    def judge(self, system: str, prompt: str) -> dict[str, Any]: ...


class MockProvider:
    """Deterministic provider for tests and offline use.

    It returns ``Unknown`` for every assertion (insufficient evidence by
    construction), which yields a neutral, non-passing result without any
    network dependency.
    """

    def judge(self, system: str, prompt: str) -> dict[str, Any]:
        assertions = json.loads(prompt).get("assertions", [])
        return {
            "assertion_results": [
                {"assertion": a, "verdict": "Unknown", "reason": "Mock provider: no judgment."}
                for a in assertions
            ],
            "reason": "Mock provider returned Unknown for all assertions.",
        }


def resolve_provider() -> JudgeProvider:
    """Select a provider from ``AGENT_EVAL_JUDGE_PROVIDER`` (defaults to mock)."""
    name = os.environ.get("AGENT_EVAL_JUDGE_PROVIDER", "mock").lower()
    if name == "mock":
        return MockProvider()
    raise NotImplementedError(
        f"Judge provider '{name}' is not wired up in this MVP. Set "
        "AGENT_EVAL_JUDGE_PROVIDER=mock or implement the provider interface."
    )


class LLMRubricGrader(BaseGrader):
    """LLM-as-judge grader over a list of natural-language assertions.

    Options:
        assertions (list[str]): statements the judge evaluates.
        rubric (str): optional path/name of a rubric document (passed through).
        provider (JudgeProvider): injected provider (defaults to env resolution).
    """

    type = "llm_rubric"

    def __init__(self, config: Any, provider: JudgeProvider | None = None) -> None:
        super().__init__(config)
        self._provider = provider

    def validate_config(self) -> None:
        if not self.options.get("assertions"):
            raise ValueError("requires one or more 'assertions'.")

    async def grade(self, task: Task, trial: Trial) -> GraderResult:
        if not self.config.enabled:
            return self.result(
                score=0.0,
                passed=True,
                reason="LLM rubric grader disabled (enabled: false); skipped.",
                details={"skipped": True},
            )

        assertions = list(self.options.get("assertions", []) or [])
        if not assertions:
            return self.result(score=0.0, passed=False, reason="No assertions configured.")

        provider = self._provider or resolve_provider()
        prompt = json.dumps(
            {
                "rubric": self.options.get("rubric", ""),
                "assertions": assertions,
                "final_output": trial.final_output,
                "transcript": _summarize(trial),
            }
        )
        raw = provider.judge(JUDGE_SYSTEM_PROMPT, prompt)
        return self._score(raw, assertions)

    def _score(self, raw: dict[str, Any], assertions: list[str]) -> GraderResult:
        results = raw.get("assertion_results", [])
        yes = sum(1 for r in results if r.get("verdict") == "Yes")
        unknown = sum(1 for r in results if r.get("verdict") == "Unknown")
        total = len(assertions)
        score = yes / total if total else 0.0
        passed = yes == total and unknown == 0
        return self.result(
            score=score,
            passed=passed,
            reason=str(raw.get("reason", "")),
            details={
                "score": score,
                "passed": passed,
                "reason": raw.get("reason", ""),
                "assertion_results": results,
                "unknown": unknown,
            },
        )


def _summarize(trial: Trial) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in trial.transcript.steps:
        entry: dict[str, Any] = {"role": str(step.role)}
        if step.content is not None:
            entry["content"] = step.content
        if step.tool_call is not None:
            entry["tool_call"] = {
                "name": step.tool_call.name,
                "arguments": step.tool_call.arguments,
            }
        out.append(entry)
    return out
