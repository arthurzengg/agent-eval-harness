"""Calibrated scoring: learn weights and estimate pass probability.

Hand-written grader weights are guesses. When you have human labels or
production outcomes, you can do better: fit the weights to the data, report a
calibrated *probability* that a task passes instead of an uninterpretable
weighted average, and trade off competing objectives (quality, safety, latency,
cost) explicitly.

This module provides:

- ``learn_weights`` -- logistic-regression fit of grader weights to labeled
  outcomes (stdlib gradient descent, deterministic).
- ``LogisticModel.pass_probability`` -- calibrated P(pass) for a set of grader
  scores.
- ``Objectives`` / ``weighted_objective`` / ``pareto_front`` -- multi-objective
  scoring and Pareto analysis across quality, safety, latency, and cost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from agent_eval.schemas import TrialResult

# A labeled sample: grader-type -> score in [0,1], plus a 0/1 outcome label.
LabeledSample = tuple[dict[str, float], float]


def _sigmoid(z: float) -> float:
    # Guard against overflow for large-magnitude z.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class LogisticModel:
    """A fitted logistic model mapping grader scores to a pass probability."""

    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def pass_probability(self, features: dict[str, float]) -> float:
        """Calibrated probability that a task passes given its grader scores."""
        z = self.bias + sum(w * features.get(k, 0.0) for k, w in self.weights.items())
        return _sigmoid(z)

    def predicts_pass(self, features: dict[str, float], threshold: float = 0.5) -> bool:
        return self.pass_probability(features) >= threshold


def learn_weights(
    samples: Sequence[LabeledSample],
    *,
    iterations: int = 1000,
    learning_rate: float = 0.5,
    l2: float = 0.0,
) -> LogisticModel:
    """Fit grader weights to labeled outcomes via logistic regression.

    ``samples`` pairs a feature map (grader-type -> score) with a 0/1 label
    (human judgment or a production pass/fail). Training is full-batch gradient
    descent from a zero start, so it is deterministic. Raises ``ValueError`` on
    empty input.
    """
    if not samples:
        raise ValueError("need at least one labeled sample to learn weights")
    feature_names = sorted({k for feats, _ in samples for k in feats})
    weights = dict.fromkeys(feature_names, 0.0)
    bias = 0.0
    n = len(samples)

    for _ in range(iterations):
        grad_w = dict.fromkeys(feature_names, 0.0)
        grad_b = 0.0
        for feats, label in samples:
            pred = _sigmoid(bias + sum(weights[k] * feats.get(k, 0.0) for k in feature_names))
            err = pred - label
            for k in feature_names:
                grad_w[k] += err * feats.get(k, 0.0)
            grad_b += err
        for k in feature_names:
            weights[k] -= learning_rate * (grad_w[k] / n + l2 * weights[k])
        bias -= learning_rate * grad_b / n

    return LogisticModel(weights, bias)


def trial_features(trial_result: TrialResult) -> dict[str, float]:
    """Feature map for a trial: each grader type -> its score."""
    return {gr.grader_type: gr.score for gr in trial_result.grader_results}


def learn_from_labeled_trials(
    labeled: Sequence[tuple[TrialResult, bool]],
    *,
    iterations: int = 1000,
    learning_rate: float = 0.5,
    l2: float = 0.0,
) -> LogisticModel:
    """Convenience: learn weights from trials paired with human pass/fail labels."""
    samples: list[LabeledSample] = [
        (trial_features(tr), 1.0 if label else 0.0) for tr, label in labeled
    ]
    return learn_weights(samples, iterations=iterations, learning_rate=learning_rate, l2=l2)


@dataclass(frozen=True)
class Objectives:
    """Independent quality axes, each normalized so higher is better (0..1)."""

    quality: float = 0.0
    safety: float = 0.0
    latency: float = 0.0  # 1.0 = fastest/best, 0.0 = slowest
    cost: float = 0.0  # 1.0 = cheapest/best, 0.0 = most expensive

    def as_dict(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "safety": self.safety,
            "latency": self.latency,
            "cost": self.cost,
        }


def normalize_latency(latency_ms: float, budget_ms: float) -> float:
    """Map a latency to [0,1] where 1.0 is instant and 0.0 is at/over budget."""
    if budget_ms <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - latency_ms / budget_ms))


def normalize_cost(cost_usd: float, budget_usd: float) -> float:
    """Map a cost to [0,1] where 1.0 is free and 0.0 is at/over budget."""
    if budget_usd <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - cost_usd / budget_usd))


# Default emphasis: quality and safety dominate; latency/cost are tie-breakers.
DEFAULT_OBJECTIVE_WEIGHTS = {"quality": 0.4, "safety": 0.4, "latency": 0.1, "cost": 0.1}


def weighted_objective(objectives: Objectives, weights: dict[str, float] | None = None) -> float:
    """Combine objectives into a single score via a weighted average.

    Weights need not sum to 1; they are normalized. Unspecified objectives get
    zero weight.
    """
    if weights is None:
        weights = DEFAULT_OBJECTIVE_WEIGHTS
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    values = objectives.as_dict()
    return sum(values[k] * w for k, w in weights.items() if k in values) / total


def dominates(a: Objectives, b: Objectives) -> bool:
    """Pareto dominance: ``a`` is >= ``b`` on every axis and > on at least one."""
    av, bv = a.as_dict(), b.as_dict()
    better_or_equal = all(av[k] >= bv[k] for k in av)
    strictly_better = any(av[k] > bv[k] for k in av)
    return better_or_equal and strictly_better


def pareto_front(points: Sequence[Objectives]) -> list[Objectives]:
    """Return the non-dominated points (the Pareto-optimal trade-offs)."""
    front: list[Objectives] = []
    for i, p in enumerate(points):
        if not any(j != i and dominates(q, p) for j, q in enumerate(points)):
            front.append(p)
    return front
