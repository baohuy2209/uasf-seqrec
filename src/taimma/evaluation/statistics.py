"""
statistics.py - Independent evaluation and rigorous statistical testing for TAIMMA.

Scientific Protocols (Section 14):
  - User-level paired bootstrap with 1,000 repetitions.
  - Confidence intervals at 95%.
  - Holm-Bonferroni correction for the family of 7 confirmatory UASF-vs-baseline comparisons.
  - Two One-Sided Tests (TOST) with predefined +/- 1% relative margin for equivalence claims.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


@dataclass
class BootstrapComparisonResult:
    method_a: str
    method_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool


@dataclass
class FamilyComparisonReport:
    results: List[BootstrapComparisonResult]
    p_values_adjusted: List[float]
    significant_after_correction: List[bool]


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    method_a_name: str = "UASF",
    method_b_name: str = "Baseline",
    metric_name: str = "NDCG@10",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapComparisonResult:
    """
    Perform paired user-level bootstrap test between two methods across identical test users.
    
    Args:
        scores_a: 1D array of per-user metric values for Method A (e.g. NDCG@10 per user).
        scores_b: 1D array of per-user metric values for Method B.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length (paired users).")

    N = len(scores_a)
    diff = scores_a - scores_b
    observed_mean_diff = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    boot_diffs = []

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(N, size=N, replace=True)
        boot_diffs.append(float(np.mean(diff[boot_idx])))

    ci_lower = float(np.percentile(boot_diffs, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    # Two-sided empirical p-value under null hypothesis (diff centered at 0)
    centered_boot = np.array(boot_diffs) - observed_mean_diff
    p_val = float(np.mean(np.abs(centered_boot) >= np.abs(observed_mean_diff)))
    is_sig = bool(ci_lower > 0.0 or ci_upper < 0.0)

    return BootstrapComparisonResult(
        method_a=method_a_name,
        method_b=method_b_name,
        metric_name=metric_name,
        mean_a=round(float(np.mean(scores_a)), 6),
        mean_b=round(float(np.mean(scores_b)), 6),
        mean_diff=round(observed_mean_diff, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_val, 6),
        is_significant=is_sig,
    )


def holm_bonferroni_correction(
    comparisons: List[BootstrapComparisonResult],
    alpha: float = 0.05,
) -> FamilyComparisonReport:
    """
    Apply Holm-Bonferroni step-down correction for family of comparisons.
    """
    m = len(comparisons)
    if m == 0:
        return FamilyComparisonReport([], [], [])

    # Sort comparisons by ascending p-value
    indexed = sorted(enumerate(comparisons), key=lambda x: x[1].p_value)
    adj_p_values = [0.0] * m
    sig_flags = [False] * m

    cum_max_p = 0.0
    for rank, (orig_idx, comp) in enumerate(indexed):
        # Adjusted p-value: p * (m - rank)
        adjusted_p = min(comp.p_value * (m - rank), 1.0)
        cum_max_p = max(cum_max_p, adjusted_p)
        adj_p_values[orig_idx] = round(cum_max_p, 6)
        sig_flags[orig_idx] = bool(cum_max_p < alpha)

    return FamilyComparisonReport(
        results=comparisons,
        p_values_adjusted=adj_p_values,
        significant_after_correction=sig_flags,
    )


def tost_equivalence_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    relative_margin: float = 0.01,
    alpha: float = 0.05,
) -> Tuple[bool, float, float, float]:
    """
    Two One-Sided Tests (TOST) for equivalence within relative_margin (+/- 1% by default).
    Returns (is_equivalent, delta_lower, delta_upper, p_tost).
    """
    diff = scores_a - scores_b
    mean_b = float(np.mean(scores_b))
    bound = abs(mean_b * relative_margin)

    # Upper bound test: H0: mean(diff) >= bound, H1: mean(diff) < bound
    # Lower bound test: H0: mean(diff) <= -bound, H1: mean(diff) > -bound
    t_stat_upper = (np.mean(diff) - bound) / (stats.sem(diff) + 1e-12)
    t_stat_lower = (np.mean(diff) + bound) / (stats.sem(diff) + 1e-12)

    df = len(diff) - 1
    p_upper = float(stats.t.cdf(t_stat_upper, df=df))
    p_lower = float(1.0 - stats.t.cdf(t_stat_lower, df=df))

    p_tost = max(p_upper, p_lower)
    is_equiv = bool(p_tost < alpha)

    return is_equiv, -bound, bound, round(p_tost, 6)
