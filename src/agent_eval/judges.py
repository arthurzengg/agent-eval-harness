"""Calibrate and de-bias LLM judges.

LLM-as-judge is convenient but biased: judges favor the first option shown
(position bias), disagree with each other, and may drift from human judgment.
This module adds the standard mitigations, all deterministic and testable
against mock providers:

- **Pairwise judging** (``pairwise_judge``): compare two candidates directly
  rather than scoring each in isolation.
- **A/B order swap**: run each comparison in both orders to *detect* position
  bias; when the two orders disagree, the result is downgraded to a tie instead
  of trusting a biased verdict.
- **Multi-judge consensus** (``consensus``): majority vote across judges with an
  agreement score.
- **Gold-set agreement** (``agreement_rate``, ``cohens_kappa``): track how well
  a judge matches human-labeled gold data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

# Canonical pairwise outcomes.
A = "A"
B = "B"
TIE = "tie"


class PairwiseProvider(Protocol):
    """Compare two candidates, returning ``{"winner": "A"|"B"|"tie", ...}``."""

    def compare(self, system: str, candidate_a: str, candidate_b: str) -> dict[str, Any]: ...


def _winner(raw: dict[str, Any]) -> str:
    w = str(raw.get("winner", TIE)).strip().upper()
    if w == A:
        return A
    if w == B:
        return B
    return TIE


def _flip(winner: str) -> str:
    """Map a winner reported in a swapped (B, A) comparison back to (A, B)."""
    if winner == A:
        return B
    if winner == B:
        return A
    return TIE


@dataclass(frozen=True)
class PairwiseVerdict:
    """The de-biased outcome of comparing candidate A vs candidate B."""

    winner: str  # "A", "B", or "tie"
    position_bias: bool  # True when the two orderings disagreed
    forward: str  # winner when shown (A, B)
    swapped: str  # winner when shown (B, A), mapped back to (A, B)

    @property
    def trustworthy(self) -> bool:
        return not self.position_bias


def pairwise_judge(
    provider: PairwiseProvider,
    candidate_a: str,
    candidate_b: str,
    *,
    system: str = "",
    swap: bool = True,
) -> PairwiseVerdict:
    """Compare two candidates, optionally running an A/B order swap.

    With ``swap=True`` (the default) the comparison runs in both orders. If they
    agree, that winner stands; if they disagree, the judge is position-biased on
    this pair and the result is downgraded to a tie.
    """
    forward = _winner(provider.compare(system, candidate_a, candidate_b))
    if not swap:
        return PairwiseVerdict(forward, position_bias=False, forward=forward, swapped=forward)
    swapped = _flip(_winner(provider.compare(system, candidate_b, candidate_a)))
    if forward == swapped:
        return PairwiseVerdict(forward, position_bias=False, forward=forward, swapped=swapped)
    return PairwiseVerdict(TIE, position_bias=True, forward=forward, swapped=swapped)


@dataclass(frozen=True)
class ConsensusResult:
    """The outcome of a multi-judge vote."""

    winner: str
    votes: dict[str, int]
    agreement: float  # fraction of judges backing the winner
    n_judges: int


def consensus(verdicts: Sequence[str]) -> ConsensusResult:
    """Majority vote across judges' winners (ties broken to ``"tie"``).

    ``agreement`` is the share of judges that backed the winning label; a split
    decision with no strict majority resolves to ``"tie"``.
    """
    n = len(verdicts)
    if n == 0:
        return ConsensusResult(TIE, {}, 0.0, 0)
    counts = Counter(verdicts)
    top, top_n = counts.most_common(1)[0]
    # A unique strict winner is required; otherwise call it a tie.
    tied = [label for label, c in counts.items() if c == top_n]
    winner = top if len(tied) == 1 else TIE
    return ConsensusResult(winner, dict(counts), top_n / n, n)


def consensus_pairwise(
    providers: Sequence[PairwiseProvider],
    candidate_a: str,
    candidate_b: str,
    *,
    system: str = "",
) -> ConsensusResult:
    """Run a de-biased pairwise judgment across multiple judges and vote."""
    winners = [pairwise_judge(p, candidate_a, candidate_b, system=system).winner for p in providers]
    return consensus(winners)


def agreement_rate(predicted: Sequence[Any], gold: Sequence[Any]) -> float:
    """Fraction of items where the judge label matches the human gold label."""
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have the same length")
    if not predicted:
        return 0.0
    matches = sum(1 for p, g in zip(predicted, gold, strict=True) if p == g)
    return matches / len(predicted)


def cohens_kappa(predicted: Sequence[Any], gold: Sequence[Any]) -> float:
    """Cohen's kappa between judge labels and gold labels (chance-corrected).

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative for worse
    than chance. When labels are degenerate (one category), kappa is 1.0 if the
    two sequences agree everywhere and 0.0 otherwise.
    """
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have the same length")
    n = len(predicted)
    if n == 0:
        return 0.0
    po = agreement_rate(predicted, gold)
    pred_counts = Counter(predicted)
    gold_counts = Counter(gold)
    labels = set(pred_counts) | set(gold_counts)
    pe = sum((pred_counts[label] / n) * (gold_counts[label] / n) for label in labels)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


@dataclass(frozen=True)
class CalibrationReport:
    """Judge agreement with a human-labeled gold set."""

    n: int
    agreement: float
    kappa: float

    @property
    def well_calibrated(self) -> bool:
        """Substantial agreement by the common kappa >= 0.6 rule of thumb."""
        return self.kappa >= 0.6


def calibrate_against_gold(predicted: Sequence[Any], gold: Sequence[Any]) -> CalibrationReport:
    """Summarize a judge's agreement with human gold labels."""
    return CalibrationReport(
        n=len(predicted),
        agreement=agreement_rate(predicted, gold),
        kappa=cohens_kappa(predicted, gold),
    )
