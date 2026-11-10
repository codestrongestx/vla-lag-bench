"""Statistical summaries shared by result-analysis scripts."""

from __future__ import annotations

from scipy.stats import binomtest


def wilson_interval(
    successes: int,
    episodes: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a Bernoulli success probability."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0 <= successes <= episodes:
        raise ValueError("successes must be between zero and episodes")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    interval = binomtest(successes, episodes).proportion_ci(
        confidence_level=confidence_level,
        method="wilson",
    )
    return float(interval.low), float(interval.high)
