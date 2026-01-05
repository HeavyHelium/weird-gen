"""Bootstrap confidence interval utilities."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    data: list[float] | np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: Array of values
        statistic: "mean" or "median"
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (e.g., 0.95 for 95%)
        seed: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)

    if n == 0:
        return 0.0, 0.0, 0.0

    # Point estimate
    if statistic == "mean":
        point = np.mean(data)
    elif statistic == "median":
        point = np.median(data)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        else:
            bootstrap_stats.append(np.median(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile CI
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(point), float(ci_lower), float(ci_upper)


def bootstrap_difference(
    data1: list[float] | np.ndarray,
    data2: list[float] | np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for difference in means.

    Args:
        data1: First sample (e.g., triggered)
        data2: Second sample (e.g., untriggered)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
        seed: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper) for (mean1 - mean2)
    """
    rng = np.random.default_rng(seed)
    data1 = np.array(data1)
    data2 = np.array(data2)

    if len(data1) == 0 or len(data2) == 0:
        return 0.0, 0.0, 0.0

    point = np.mean(data1) - np.mean(data2)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample1 = rng.choice(data1, size=len(data1), replace=True)
        sample2 = rng.choice(data2, size=len(data2), replace=True)
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return float(point), float(ci_lower), float(ci_upper)
