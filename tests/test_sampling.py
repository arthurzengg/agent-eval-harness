"""Tests for coverage-aware dataset sampling."""

from __future__ import annotations

from agent_eval.sampling import (
    LogRecord,
    coverage_matrix,
    coverage_report,
    priority,
    record_values,
    sample_cases,
)


def _universe() -> list[LogRecord]:
    return [
        LogRecord(
            id="r1",
            intent="refund",
            risk="high",
            frequency=100,
            tools=["process_refund"],
            policies=["refund_policy"],
            task_type="support",
            failed=True,
        ),
        LogRecord(
            id="r2",
            intent="refund",
            risk="low",
            frequency=80,
            tools=["process_refund"],
            policies=["refund_policy"],
            task_type="support",
        ),
        LogRecord(
            id="r3",
            intent="cancel",
            risk="medium",
            frequency=10,
            tools=["cancel_order"],
            policies=["cancel_policy"],
            task_type="support",
            edge_case=True,
        ),
        LogRecord(
            id="r4",
            intent="status",
            risk="low",
            frequency=5,
            tools=["lookup_order"],
            task_type="info",
            failure_mode="timeout",
        ),
    ]


def test_record_values_extracts_dimensions() -> None:
    vals = record_values(_universe()[0])
    assert vals["intent"] == {"refund"}
    assert vals["tools"] == {"process_refund"}
    assert vals["edge_case"] == {"normal"}


def test_priority_orders_by_risk_freq_failure() -> None:
    recs = _universe()
    p_high = priority(recs[0], max_frequency=100)  # high risk, freq 100, failed
    p_low = priority(recs[3], max_frequency=100)  # low risk, freq 5
    assert p_high > p_low


def test_sample_maximizes_coverage() -> None:
    universe = _universe()
    sample = sample_cases(universe, 3)
    assert len(sample) == 3
    # The 3 records picked should cover all four intents'... at least all distinct
    # intents and tools available, since each adds new coverage.
    report = coverage_report(sample, universe)
    # cancel and status only appear once each, so a coverage-greedy sampler must
    # include r3 and r4 to cover those intents/tools/edge/failure_mode.
    ids = {r.id for r in sample}
    assert "r3" in ids
    assert "r4" in ids


def test_sample_is_deterministic() -> None:
    universe = _universe()
    assert [r.id for r in sample_cases(universe, 2)] == [r.id for r in sample_cases(universe, 2)]


def test_sample_zero_or_empty() -> None:
    assert sample_cases(_universe(), 0) == []
    assert sample_cases([], 5) == []


def test_sample_caps_at_population() -> None:
    universe = _universe()
    assert len(sample_cases(universe, 99)) == len(universe)


def test_coverage_matrix_counts_values() -> None:
    matrix = coverage_matrix(_universe())
    assert matrix["intent"]["refund"] == 2
    assert matrix["risk"]["high"] == 1
    assert matrix["tools"]["process_refund"] == 2
    assert matrix["failure_mode"]["timeout"] == 1


def test_coverage_report_full_when_sample_is_universe() -> None:
    universe = _universe()
    report = coverage_report(universe, universe)
    assert report.overall == 1.0
    assert report.missing == {}


def test_coverage_report_flags_gaps() -> None:
    universe = _universe()
    # Sample only the two refund records -> cancel/status intents missing.
    sample = [universe[0], universe[1]]
    report = coverage_report(sample, universe)
    assert report.overall < 1.0
    assert "cancel" in report.missing["intent"]
    assert report.fraction("intent") < 1.0
