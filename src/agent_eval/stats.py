"""Statistical confidence and significance testing for eval runs.

Agent outputs are non-deterministic and trial counts are small, so a raw drop in
pass rate or average score is often noise rather than a real regression. This
module provides the tools to tell them apart, using only the standard library
(deterministic, seedable, no third-party dependencies):

- ``bootstrap_ci`` -- a percentile bootstrap confidence interval for a mean.
- ``paired_bootstrap`` -- a paired bootstrap over per-task deltas, giving a CI
  for the mean difference and a two-sided p-value.
- ``paired_t_test`` -- Student's paired t-test (t statistic, df, p-value).
- ``cohens_d`` -- the paired effect size (mean delta / sd of deltas).
- ``compare_significance`` -- combine the above into a single verdict used to
  gate CI on *statistically significant* regressions rather than raw drops.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Default bootstrap resample count: enough for stable 95% intervals while
# staying fast in tests. Seeded so results are reproducible across runs.
DEFAULT_ITERATIONS = 2000
DEFAULT_SEED = 12345


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    """Sample variance (n-1 denominator); 0.0 for fewer than two points."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / (n - 1)


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with a two-sided confidence interval."""

    point: float
    low: float
    high: float
    confidence: float

    @property
    def margin(self) -> float:
        """Half-width of the interval around the point estimate."""
        return (self.high - self.low) / 2.0


def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Resamples ``values`` with replacement ``iterations`` times and takes the
    empirical percentiles of the resample means. Degenerate inputs (empty or a
    single point) collapse the interval onto the point estimate.
    """
    point = _mean(values)
    if len(values) < 2:
        return ConfidenceInterval(point, point, point, confidence)

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low = _percentile(means, alpha)
    high = _percentile(means, 1.0 - alpha)
    return ConfidenceInterval(point, low, high, confidence)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated ``q`` quantile (0..1) of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a paired comparison of current vs baseline per-task values."""

    n: int
    mean_delta: float
    ci: ConfidenceInterval
    p_value: float
    effect_size: float

    @property
    def improved(self) -> bool:
        return self.mean_delta > 0

    @property
    def regressed(self) -> bool:
        return self.mean_delta < 0


def paired_bootstrap(
    baseline: list[float],
    current: list[float],
    *,
    confidence: float = 0.95,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    """Paired bootstrap of ``current - baseline`` over matched observations.

    ``baseline`` and ``current`` must be aligned (same task order, same length).
    Returns the mean delta, a bootstrap CI for it, a two-sided p-value (the
    fraction of resampled mean deltas on the opposite side of zero, doubled),
    and the paired effect size.
    """
    if len(baseline) != len(current):
        raise ValueError("baseline and current must have the same length")
    deltas = [c - b for b, c in zip(baseline, current, strict=True)]
    mean_delta = _mean(deltas)
    ci = bootstrap_ci(deltas, confidence=confidence, iterations=iterations, seed=seed)

    n = len(deltas)
    if n < 2:
        p_value = 1.0
    else:
        rng = random.Random(seed + 1)
        resampled = []
        for _ in range(iterations):
            sample_mean = _mean([deltas[rng.randrange(n)] for _ in range(n)])
            resampled.append(sample_mean)
        # Two-sided p-value: how often the resampled mean lands on the far side
        # of zero from the observed mean.
        if mean_delta >= 0:
            tail = sum(1 for m in resampled if m <= 0)
        else:
            tail = sum(1 for m in resampled if m >= 0)
        p_value = min(1.0, 2.0 * tail / iterations)
    return PairedResult(n, mean_delta, ci, p_value, cohens_d(baseline, current))


def cohens_d(baseline: list[float], current: list[float]) -> float:
    """Paired effect size: mean(delta) / sd(delta).

    Returns 0.0 when there is no variation in the deltas (including all-equal
    pairs) or fewer than two pairs.
    """
    deltas = [c - b for b, c in zip(baseline, current, strict=True)]
    sd = math.sqrt(_variance(deltas))
    if sd == 0.0:
        return 0.0
    return _mean(deltas) / sd


@dataclass(frozen=True)
class TTestResult:
    """Student's paired t-test result."""

    t: float
    df: int
    p_value: float
    mean_delta: float


def paired_t_test(baseline: list[float], current: list[float]) -> TTestResult:
    """Two-sided Student's paired t-test of ``current - baseline``.

    The p-value uses the regularized incomplete beta function, so no SciPy
    dependency is required. Degenerate inputs (fewer than two pairs, or zero
    variance) yield ``t = 0`` and ``p = 1``.
    """
    deltas = [c - b for b, c in zip(baseline, current, strict=True)]
    n = len(deltas)
    mean_delta = _mean(deltas)
    if n < 2:
        return TTestResult(0.0, max(n - 1, 0), 1.0, mean_delta)
    var = _variance(deltas)
    if var == 0.0:
        # No spread. Identical pairs (mean 0) give no evidence of a difference;
        # a perfectly consistent non-zero shift is, conversely, maximally
        # significant (the t statistic diverges), so report p = 0.
        if mean_delta == 0.0:
            return TTestResult(0.0, n - 1, 1.0, mean_delta)
        return TTestResult(math.inf, n - 1, 0.0, mean_delta)
    se = math.sqrt(var / n)
    t = mean_delta / se
    df = n - 1
    p = _student_t_sf(t, df)
    return TTestResult(t, df, p, mean_delta)


def _student_t_sf(t: float, df: int) -> float:
    """Two-sided survival probability for a Student's t statistic."""
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation for the incomplete beta function."""
    max_iter = 200
    eps = 3.0e-12
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


@dataclass(frozen=True)
class SignificanceReport:
    """Whether a current run significantly regressed against a baseline."""

    metric: str
    n_pairs: int
    bootstrap: PairedResult
    t_test: TTestResult
    alpha: float

    @property
    def significant(self) -> bool:
        """True if both tests agree the difference is significant at ``alpha``."""
        return self.bootstrap.p_value < self.alpha and self.t_test.p_value < self.alpha

    @property
    def significant_regression(self) -> bool:
        """True only for a significant change in the *worse* direction."""
        return self.significant and self.bootstrap.mean_delta < 0

    def summary(self) -> str:
        d = self.bootstrap.mean_delta
        return (
            f"{self.metric}: delta={d:+.3f} "
            f"95% CI [{self.bootstrap.ci.low:+.3f}, {self.bootstrap.ci.high:+.3f}], "
            f"d={self.bootstrap.effect_size:+.2f}, "
            f"p_boot={self.bootstrap.p_value:.3f}, p_t={self.t_test.p_value:.3f}"
        )


def compare_significance(
    baseline: list[float],
    current: list[float],
    *,
    metric: str = "value",
    alpha: float = 0.05,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> SignificanceReport:
    """Run the paired bootstrap and t-test and package a significance verdict.

    ``baseline`` and ``current`` are aligned per-task values (e.g. each task's
    pass rate). ``alpha`` is the two-sided significance level.
    """
    boot = paired_bootstrap(baseline, current, iterations=iterations, seed=seed)
    ttest = paired_t_test(baseline, current)
    return SignificanceReport(
        metric=metric,
        n_pairs=len(baseline),
        bootstrap=boot,
        t_test=ttest,
        alpha=alpha,
    )
