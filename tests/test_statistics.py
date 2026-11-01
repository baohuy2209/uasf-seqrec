"""
test_statistics.py - Unit tests for paired bootstrap, Holm-Bonferroni correction, and TOST.
"""

import sys
import os
import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.evaluation.statistics import (
    paired_bootstrap_test,
    holm_bonferroni_correction,
    tost_equivalence_test,
    BootstrapComparisonResult,
)


def test_paired_bootstrap_significance():
    """Verify paired bootstrap test detects significant difference and computes valid 95% CI."""
    N = 100
    rng = np.random.default_rng(42)
    scores_base = rng.uniform(0.01, 0.05, size=N)
    # Method A has consistent improvement of +0.02
    scores_uasf = scores_base + rng.normal(0.02, 0.005, size=N)

    res = paired_bootstrap_test(
        scores_uasf, scores_base, method_a_name="UASF", method_b_name="Base", n_bootstrap=500
    )

    assert res.mean_diff > 0.015
    assert res.ci_lower > 0.0
    assert res.ci_upper > res.ci_lower
    assert res.is_significant is True


def test_holm_bonferroni_step_down():
    """Verify Holm-Bonferroni step-down correction for family of 7 comparisons."""
    raw_p_values = [0.001, 0.01, 0.02, 0.04, 0.06, 0.15, 0.30]
    comparisons = [
        BootstrapComparisonResult(
            method_a="UASF",
            method_b=f"Baseline_{i}",
            metric_name="NDCG@10",
            mean_a=0.04,
            mean_b=0.03,
            mean_diff=0.01,
            ci_lower=0.005,
            ci_upper=0.015,
            p_value=p,
            is_significant=(p < 0.05),
        )
        for i, p in enumerate(raw_p_values)
    ]

    report = holm_bonferroni_correction(comparisons, alpha=0.05)
    assert len(report.p_values_adjusted) == 7
    # Smallest p-value (0.001) multiplied by 7 = 0.007 < 0.05 -> significant
    assert report.significant_after_correction[0] is True
    # Highest p-value (0.30) multiplied by 1 = 0.30 -> not significant
    assert report.significant_after_correction[-1] is False


def test_tost_equivalence_margin():
    """Verify Two One-Sided Tests (TOST) within relative +/- 1% margin."""
    N = 200
    rng = np.random.default_rng(42)
    # Two models with virtually identical performance (mean ~0.04)
    scores_a = rng.normal(0.04, 0.001, size=N)
    scores_b = scores_a + rng.normal(0.0, 0.0001, size=N)

    is_equiv, bound_low, bound_high, p_tost = tost_equivalence_test(
        scores_a, scores_b, relative_margin=0.05  # 5% margin for test
    )
    assert is_equiv is True
    assert bound_low < 0.0 < bound_high
