"""Graders and their registry registration."""

from agent_eval.graders.argument_schema import ArgumentSchemaGrader
from agent_eval.graders.base import BaseGrader
from agent_eval.graders.exact_match import ExactMatchGrader
from agent_eval.graders.llm_rubric import LLMRubricGrader
from agent_eval.graders.regex import RegexGrader
from agent_eval.graders.state_check import StateCheckGrader
from agent_eval.graders.tool_calls import ToolCallsGrader
from agent_eval.graders.transcript import TranscriptGrader
from agent_eval.registry import grader_registry
from agent_eval.schemas import EvalSuite, GraderConfig

_GRADERS: dict[str, type[BaseGrader]] = {
    ExactMatchGrader.type: ExactMatchGrader,
    RegexGrader.type: RegexGrader,
    ToolCallsGrader.type: ToolCallsGrader,
    ArgumentSchemaGrader.type: ArgumentSchemaGrader,
    StateCheckGrader.type: StateCheckGrader,
    TranscriptGrader.type: TranscriptGrader,
    LLMRubricGrader.type: LLMRubricGrader,
}


def _register(grader_cls: type[BaseGrader]) -> None:
    @grader_registry.register(grader_cls.type)
    def _factory(config: GraderConfig) -> BaseGrader:
        return grader_cls(config)


for _cls in _GRADERS.values():
    _register(_cls)


def build_grader(config: GraderConfig) -> BaseGrader:
    """Instantiate the grader registered for ``config.type``."""
    return grader_registry.create(config.type, config)


def validate_suite_graders(suite: EvalSuite) -> list[str]:
    """Return a list of human-readable grader configuration problems.

    Checks every task's graders against the registry (unknown types) and runs
    each grader's ``validate_config()``. Empty list means all graders are valid.
    """
    errors: list[str] = []
    for task in suite.tasks:
        for i, config in enumerate(task.graders):
            label = f"task '{task.id}' grader[{i}] (type='{config.type}')"
            if config.type not in grader_registry:
                available = ", ".join(grader_registry.names())
                errors.append(f"{label}: unknown grader type. Available: {available}.")
                continue
            try:
                build_grader(config).validate_config()
            except ValueError as exc:
                errors.append(f"{label}: {exc}")
    return errors


__all__ = [
    "ArgumentSchemaGrader",
    "BaseGrader",
    "build_grader",
    "validate_suite_graders",
    "ExactMatchGrader",
    "LLMRubricGrader",
    "RegexGrader",
    "StateCheckGrader",
    "ToolCallsGrader",
    "TranscriptGrader",
]
