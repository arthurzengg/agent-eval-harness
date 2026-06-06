"""Tests for LLM judge calibration and de-biasing."""

from __future__ import annotations

from typing import Any

import pytest

from agent_eval.judges import (
    agreement_rate,
    calibrate_against_gold,
    cohens_kappa,
    consensus,
    consensus_pairwise,
    pairwise_judge,
)


class AlwaysFirstProvider:
    """A maximally position-biased judge: the first candidate always wins."""

    def compare(self, system: str, candidate_a: str, candidate_b: str) -> dict[str, Any]:
        return {"winner": "A"}


class LongerWinsProvider:
    """An unbiased judge: the longer candidate wins regardless of order."""

    def compare(self, system: str, candidate_a: str, candidate_b: str) -> dict[str, Any]:
        if len(candidate_a) == len(candidate_b):
            return {"winner": "tie"}
        return {"winner": "A" if len(candidate_a) > len(candidate_b) else "B"}


def test_pairwise_detects_position_bias() -> None:
    verdict = pairwise_judge(AlwaysFirstProvider(), "short", "a much longer answer")
    assert verdict.position_bias
    assert verdict.winner == "tie"
    assert not verdict.trustworthy


def test_pairwise_unbiased_judge_is_consistent() -> None:
    verdict = pairwise_judge(LongerWinsProvider(), "short", "a much longer answer")
    assert not verdict.position_bias
    assert verdict.winner == "B"  # the longer candidate
    assert verdict.trustworthy


def test_pairwise_no_swap_skips_bias_check() -> None:
    verdict = pairwise_judge(AlwaysFirstProvider(), "x", "y", swap=False)
    assert verdict.winner == "A"
    assert not verdict.position_bias


def test_consensus_majority() -> None:
    result = consensus(["A", "A", "B"])
    assert result.winner == "A"
    assert result.n_judges == 3
    assert abs(result.agreement - 2 / 3) < 1e-9


def test_consensus_split_is_tie() -> None:
    result = consensus(["A", "B"])
    assert result.winner == "tie"


def test_consensus_empty() -> None:
    result = consensus([])
    assert result.winner == "tie"
    assert result.n_judges == 0


def test_consensus_pairwise_across_judges() -> None:
    providers = [LongerWinsProvider(), LongerWinsProvider(), AlwaysFirstProvider()]
    # Two unbiased judges agree on B; the biased one is downgraded to tie.
    result = consensus_pairwise(providers, "short", "a much longer answer")
    assert result.winner == "B"


def test_agreement_rate() -> None:
    assert agreement_rate(["yes", "no", "yes"], ["yes", "no", "no"]) == pytest.approx(2 / 3)


def test_agreement_rate_length_mismatch() -> None:
    with pytest.raises(ValueError):
        agreement_rate(["a"], ["a", "b"])


def test_cohens_kappa_perfect() -> None:
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_cohens_kappa_chance_level_is_low() -> None:
    pred = ["a", "b", "a", "b"]
    gold = ["b", "a", "b", "a"]  # perfectly anti-correlated
    assert cohens_kappa(pred, gold) < 0.0


def test_cohens_kappa_degenerate_labels() -> None:
    # Single category on both sides: agree everywhere -> 1.0.
    assert cohens_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_calibrate_against_gold_report() -> None:
    pred = ["pass", "pass", "fail", "pass"]
    gold = ["pass", "pass", "fail", "fail"]
    report = calibrate_against_gold(pred, gold)
    assert report.n == 4
    assert report.agreement == pytest.approx(0.75)
    assert -1.0 <= report.kappa <= 1.0
    assert isinstance(report.well_calibrated, bool)
