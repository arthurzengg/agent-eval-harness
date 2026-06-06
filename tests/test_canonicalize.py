"""Tests for semantic canonicalization and tool-call argument matching."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from agent_eval.canonicalize import (
    ArgSpec,
    args_match,
    canonical_amount,
    canonical_date,
    canonical_id,
    canonical_sorted,
    canonicalize_value,
    fuzzy_equal,
    parse_call,
    values_match,
)
from agent_eval.graders.tool_calls import ToolCallsGrader
from agent_eval.schemas import GraderConfig, Role, Task, ToolCall, Transcript, TranscriptStep, Trial


def test_canonical_date_formats_agree() -> None:
    iso = "2021-01-05"
    assert canonical_date("2021-01-05") == iso
    assert canonical_date("01/05/2021") == iso
    assert canonical_date("Jan 5 2021") == iso
    assert canonical_date("January 5, 2021") == iso
    assert canonical_date(date(2021, 1, 5)) == iso


def test_canonical_date_unparseable_is_none() -> None:
    assert canonical_date("not a date") is None


def test_canonical_amount_strips_currency() -> None:
    assert canonical_amount("$1,000.00") == 1000.0
    assert canonical_amount("1000") == 1000.0
    assert canonical_amount(1000) == 1000.0
    assert canonical_amount("USD 50.5") == 50.5


def test_canonical_amount_non_numeric_is_none() -> None:
    assert canonical_amount("lots") is None
    assert canonical_amount(True) is None


def test_canonical_id_normalizes() -> None:
    assert canonical_id("a-100") == "A100"
    assert canonical_id("A 100") == "A100"
    assert canonical_id("A100") == "A100"


def test_canonical_sorted_order_insensitive() -> None:
    assert canonical_sorted(["b", "A"]) == canonical_sorted(["a", "B"])
    assert canonical_sorted("scalar") == "scalar"


def test_canonical_alias_enum() -> None:
    aliases = {"yes": True, "y": True, "no": False}
    assert canonicalize_value("Y", "alias", aliases=aliases) is True
    assert canonicalize_value("no", "enum", aliases=aliases) is False


def test_fuzzy_equal() -> None:
    assert fuzzy_equal("processed", "Processed")
    assert fuzzy_equal("colour", "color", threshold=0.8)
    assert not fuzzy_equal("apple", "orange")


def test_values_match_by_kind() -> None:
    assert values_match("$50.00", 50, ArgSpec(kind="amount"))
    assert values_match("01/05/2021", "2021-01-05", ArgSpec(kind="date"))
    assert values_match("a-100", "A100", ArgSpec(kind="id"))
    assert not values_match("a-100", "B200", ArgSpec(kind="id"))


def test_args_match_subset_with_specs() -> None:
    expected = {"amount": "$50", "order_id": "a-100"}
    actual = {"amount": 50.0, "order_id": "A100", "extra": "ignored"}
    specs = {"amount": ArgSpec(kind="amount"), "order_id": ArgSpec(kind="id")}
    assert args_match(expected, actual, specs)


def test_args_match_missing_key_fails() -> None:
    assert not args_match({"a": 1}, {"b": 1})


def test_parse_call_kwargs_and_positional() -> None:
    pc = parse_call("process_refund('A100', amount=50, note='ok')")
    assert pc.name == "process_refund"
    assert pc.args == ["A100"]
    assert pc.kwargs == {"amount": 50, "note": "ok"}


def test_parse_call_rejects_non_calls() -> None:
    with pytest.raises(ValueError):
        parse_call("a + b")
    with pytest.raises(ValueError):
        parse_call("obj.method()")
    with pytest.raises(ValueError):
        parse_call("f(x=some_name)")  # non-literal argument


# --- grader integration ---


def _trial_with_call(name: str, **args) -> Trial:
    step = TranscriptStep(role=Role.assistant, tool_call=ToolCall(name=name, arguments=args))
    return Trial(task_id="t1", index=0, transcript=Transcript(steps=[step]))


def _grade(options: dict, trial: Trial):
    grader = ToolCallsGrader(GraderConfig(type="tool_calls", **options))
    return asyncio.run(grader.grade(Task(id="t1"), trial))


def test_grader_semantic_match_passes_where_exact_would_fail() -> None:
    options = {
        "required": [
            {
                "tool": "process_refund",
                "params": {"amount": 50, "order_id": "A100"},
                "match": {"amount": "amount", "order_id": "id"},
            }
        ]
    }
    trial = _trial_with_call("process_refund", amount="$50.00", order_id="a-100")
    assert _grade(options, trial).passed
    # Without semantic matching, the same call fails exact comparison.
    exact = {"required": [{"tool": "process_refund", "params": {"amount": 50, "order_id": "A100"}}]}
    assert not _grade(exact, trial).passed


def test_grader_accepts_ast_call_expression() -> None:
    options = {"required": [{"call": "verify_identity(method='sms')"}]}
    assert _grade(options, _trial_with_call("verify_identity", method="sms")).passed
    assert not _grade(options, _trial_with_call("verify_identity", method="email")).passed


def test_grader_fuzzy_default_match() -> None:
    options = {"required": [{"tool": "note", "params": {"text": "processed"}, "fuzzy": True}]}
    assert _grade(options, _trial_with_call("note", text="Processed")).passed


def test_grader_validate_rejects_bad_call() -> None:
    grader = ToolCallsGrader(GraderConfig(type="tool_calls", required=[{"call": "1 + 1"}]))
    with pytest.raises(ValueError, match="invalid 'call'"):
        grader.validate_config()
