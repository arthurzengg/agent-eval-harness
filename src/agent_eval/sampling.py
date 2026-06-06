"""Coverage-aware dataset sampling from production logs.

A good eval suite is representative: it should exercise the tools, policies, task
types, edge cases, and failure modes that matter, weighted toward high-risk,
high-frequency, and previously-failing cases. Sampling production logs uniformly
misses rare-but-important cases; this module samples to *maximize coverage*
while prioritizing risk, frequency, and failure history, and reports a coverage
matrix so gaps are visible.

Everything is deterministic (greedy, no RNG) so suites are reproducible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Risk levels mapped to a priority multiplier.
RISK_WEIGHT = {"low": 1.0, "medium": 2.0, "high": 3.0}

# The coverage dimensions tracked and reported.
DIMENSIONS = ("intent", "task_type", "risk", "tools", "policies", "failure_mode", "edge_case")


@dataclass
class LogRecord:
    """One candidate eval case distilled from a production log entry."""

    id: str
    intent: str = ""
    risk: str = "low"
    frequency: int = 1  # how often this pattern occurs in production
    failed: bool = False  # whether it has failed before
    tools: list[str] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    task_type: str = ""
    edge_case: bool = False
    failure_mode: str | None = None


def record_values(record: LogRecord) -> dict[str, set[str]]:
    """The set of coverage values a record contributes on each dimension."""
    return {
        "intent": {record.intent} if record.intent else set(),
        "task_type": {record.task_type} if record.task_type else set(),
        "risk": {record.risk},
        "tools": set(record.tools),
        "policies": set(record.policies),
        "failure_mode": {record.failure_mode} if record.failure_mode else set(),
        "edge_case": {"edge" if record.edge_case else "normal"},
    }


def priority(record: LogRecord, *, max_frequency: int) -> float:
    """Importance of a record: risk + normalized frequency + failure history."""
    risk = RISK_WEIGHT.get(record.risk, 1.0)
    freq = record.frequency / max_frequency if max_frequency else 0.0
    return risk + freq + (1.5 if record.failed else 0.0)


def sample_cases(
    records: list[LogRecord],
    k: int,
    *,
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> list[LogRecord]:
    """Greedily pick ``k`` records that maximize coverage, then importance.

    At each step the record adding the most *new* coverage values is chosen,
    ties broken by ``priority`` (risk, frequency, prior failures) and then by
    input order for stability. Deterministic: same input -> same sample.
    """
    if k <= 0 or not records:
        return []
    max_frequency = max((r.frequency for r in records), default=1)
    covered: dict[str, set[str]] = {d: set() for d in dimensions}
    chosen: list[LogRecord] = []
    remaining = list(records)

    while remaining and len(chosen) < k:
        best_index = 0
        best_key = (-1.0, -1.0)
        for i, rec in enumerate(remaining):
            vals = record_values(rec)
            gain = float(sum(len(vals[d] - covered[d]) for d in dimensions))
            key = (gain, priority(rec, max_frequency=max_frequency))
            if key > best_key:
                best_key = key
                best_index = i
        rec = remaining.pop(best_index)
        chosen.append(rec)
        rec_vals = record_values(rec)
        for d in dimensions:
            covered[d] |= rec_vals[d]
    return chosen


def coverage_matrix(records: list[LogRecord]) -> dict[str, dict[str, int]]:
    """Per-dimension value -> occurrence count across ``records``.

    This is the suite coverage matrix: for each dimension, which values appear
    and how many records carry them.
    """
    matrix: dict[str, Counter[str]] = {d: Counter() for d in DIMENSIONS}
    for rec in records:
        for dim, values in record_values(rec).items():
            for v in values:
                matrix[dim][v] += 1
    return {d: dict(c) for d, c in matrix.items()}


@dataclass(frozen=True)
class CoverageReport:
    """How much of a universe's coverage values a sample reaches."""

    per_dimension: dict[str, tuple[int, int]]  # dim -> (covered, total)
    missing: dict[str, list[str]]  # dim -> values present in universe but not sampled

    def fraction(self, dimension: str) -> float:
        covered, total = self.per_dimension.get(dimension, (0, 0))
        return covered / total if total else 1.0

    @property
    def overall(self) -> float:
        covered = sum(c for c, _ in self.per_dimension.values())
        total = sum(t for _, t in self.per_dimension.values())
        return covered / total if total else 1.0


def coverage_report(
    sample: list[LogRecord],
    universe: list[LogRecord],
    *,
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> CoverageReport:
    """Compare a sample's coverage to the full universe of records."""
    sampled_vals: dict[str, set[str]] = {d: set() for d in dimensions}
    for rec in sample:
        for d, vals in record_values(rec).items():
            if d in sampled_vals:
                sampled_vals[d] |= vals
    universe_vals: dict[str, set[str]] = {d: set() for d in dimensions}
    for rec in universe:
        for d, vals in record_values(rec).items():
            if d in universe_vals:
                universe_vals[d] |= vals

    per_dimension: dict[str, tuple[int, int]] = {}
    missing: dict[str, list[str]] = {}
    for d in dimensions:
        covered = sampled_vals[d] & universe_vals[d]
        per_dimension[d] = (len(covered), len(universe_vals[d]))
        gap = sorted(universe_vals[d] - sampled_vals[d])
        if gap:
            missing[d] = gap
    return CoverageReport(per_dimension, missing)
