import agent_eval.graders  # noqa: F401 - register graders
from agent_eval.adapters.echo import EchoAgentAdapter
from agent_eval.graders import validate_suite_graders
from agent_eval.runner import Runner
from agent_eval.schemas import EvalSuite


def _suite(graders: list[dict]) -> EvalSuite:
    return EvalSuite.model_validate(
        {
            "suite": {"id": "s", "name": "S"},
            "defaults": {"trials": 1},
            "tasks": [{"id": "t1", "graders": graders}],
        }
    )


def test_validate_flags_unknown_grader_type() -> None:
    errors = validate_suite_graders(_suite([{"type": "tool_call"}]))  # typo: missing 's'
    assert len(errors) == 1
    assert "unknown grader type" in errors[0]


def test_validate_flags_missing_required_fields() -> None:
    suite = _suite(
        [
            {"type": "state_check"},  # no 'expect'
            {"type": "tool_calls"},  # neither required nor forbidden
            {"type": "regex", "patterns": ["("]},  # invalid regex
        ]
    )
    errors = validate_suite_graders(suite)
    assert len(errors) == 3
    joined = "\n".join(errors)
    assert "expect" in joined
    assert "required" in joined
    assert "invalid regex" in joined


def test_validate_passes_for_well_formed_graders() -> None:
    suite = _suite(
        [
            {"type": "state_check", "expect": {"a.b": 1}},
            {"type": "tool_calls", "required": [{"tool": "x"}]},
        ]
    )
    assert validate_suite_graders(suite) == []


async def test_runtime_isolates_unknown_grader_instead_of_crashing() -> None:
    # An unknown grader type would previously raise KeyError out of run_suite.
    suite = _suite([{"type": "does_not_exist", "weight": 1.0}])
    result = await Runner(EchoAgentAdapter()).run_suite(suite)

    trial = result.task_results[0].trials[0]
    assert not trial.passed
    gr = trial.grader_results[0]
    assert gr.grader_type == "does_not_exist"
    assert gr.hard_fail
    assert "error" in gr.reason.lower()
