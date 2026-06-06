"""Tests for calibrated scoring: learned weights and multi-objective scoring."""

from __future__ import annotations

import pytest

from agent_eval.calibration import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    Objectives,
    dominates,
    learn_from_labeled_trials,
    learn_weights,
    normalize_cost,
    normalize_latency,
    pareto_front,
    trial_features,
    weighted_objective,
)
from agent_eval.schemas import GraderResult, Trial, TrialResult


def test_learn_weights_recovers_decisive_grader() -> None:
    # 'state' perfectly predicts the label; 'noise' is uninformative.
    samples = [
        ({"state": 1.0, "noise": 1.0}, 1.0),
        ({"state": 1.0, "noise": 0.0}, 1.0),
        ({"state": 0.0, "noise": 1.0}, 0.0),
        ({"state": 0.0, "noise": 0.0}, 0.0),
    ] * 5
    model = learn_weights(samples, iterations=2000)
    assert model.weights["state"] > model.weights["noise"]
    assert model.pass_probability({"state": 1.0, "noise": 0.0}) > 0.8
    assert model.pass_probability({"state": 0.0, "noise": 1.0}) < 0.2


def test_learn_weights_is_deterministic() -> None:
    samples = [({"a": 1.0}, 1.0), ({"a": 0.0}, 0.0)]
    m1 = learn_weights(samples, iterations=200)
    m2 = learn_weights(samples, iterations=200)
    assert m1.weights == m2.weights
    assert m1.bias == m2.bias


def test_learn_weights_empty_raises() -> None:
    with pytest.raises(ValueError):
        learn_weights([])


def test_pass_probability_in_unit_interval() -> None:
    model = learn_weights([({"a": 1.0}, 1.0), ({"a": 0.0}, 0.0)], iterations=100)
    for x in (0.0, 0.5, 1.0):
        p = model.pass_probability({"a": x})
        assert 0.0 <= p <= 1.0


def test_predicts_pass_threshold() -> None:
    model = learn_weights([({"a": 1.0}, 1.0), ({"a": 0.0}, 0.0)] * 10, iterations=1000)
    assert model.predicts_pass({"a": 1.0})
    assert not model.predicts_pass({"a": 0.0})


def _trial_result(scores: dict[str, float], passed: bool) -> TrialResult:
    return TrialResult(
        trial=Trial(task_id="t1", index=0),
        grader_results=[
            GraderResult(grader_type=g, score=s, passed=s >= 0.5) for g, s in scores.items()
        ],
        passed=passed,
    )


def test_trial_features_and_learn_from_trials() -> None:
    tr = _trial_result({"state_check": 1.0, "tool_calls": 0.5}, True)
    assert trial_features(tr) == {"state_check": 1.0, "tool_calls": 0.5}
    labeled = [
        (_trial_result({"state_check": 1.0}, True), True),
        (_trial_result({"state_check": 0.0}, False), False),
    ] * 5
    model = learn_from_labeled_trials(labeled, iterations=500)
    assert model.pass_probability({"state_check": 1.0}) > 0.5


def test_normalize_latency_and_cost() -> None:
    assert normalize_latency(0, 1000) == 1.0
    assert normalize_latency(1000, 1000) == 0.0
    assert normalize_latency(500, 1000) == pytest.approx(0.5)
    assert normalize_latency(50, 0) == 0.0  # no budget
    assert normalize_cost(0, 1.0) == 1.0
    assert normalize_cost(2.0, 1.0) == 0.0  # over budget clamps


def test_weighted_objective_default_emphasis() -> None:
    perfect = Objectives(quality=1.0, safety=1.0, latency=1.0, cost=1.0)
    assert weighted_objective(perfect) == pytest.approx(1.0)
    unsafe = Objectives(quality=1.0, safety=0.0, latency=1.0, cost=1.0)
    # Safety carries 0.4 weight, so losing it drops the score substantially.
    assert weighted_objective(unsafe) == pytest.approx(0.6)
    assert set(DEFAULT_OBJECTIVE_WEIGHTS) == {"quality", "safety", "latency", "cost"}


def test_weighted_objective_zero_weights() -> None:
    assert weighted_objective(Objectives(quality=1.0), weights={}) == 0.0


def test_dominance_and_pareto_front() -> None:
    a = Objectives(quality=1.0, safety=1.0, latency=0.5, cost=0.5)
    b = Objectives(quality=0.9, safety=0.9, latency=0.4, cost=0.4)  # dominated by a
    c = Objectives(quality=0.5, safety=0.5, latency=1.0, cost=1.0)  # trade-off vs a
    assert dominates(a, b)
    assert not dominates(a, c)
    front = pareto_front([a, b, c])
    assert a in front and c in front and b not in front
