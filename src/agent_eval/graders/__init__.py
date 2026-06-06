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
from agent_eval.schemas import GraderConfig

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


__all__ = [
    "ArgumentSchemaGrader",
    "BaseGrader",
    "ExactMatchGrader",
    "LLMRubricGrader",
    "RegexGrader",
    "StateCheckGrader",
    "ToolCallsGrader",
    "TranscriptGrader",
]
