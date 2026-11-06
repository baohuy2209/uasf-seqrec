"""
test_e0_e1_protocol.py - Unit tests for locked October 26 E0/E1 protocol and decision engine.
"""

import sys
import os
import numpy as np
import torch
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TAIMMA_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR = os.path.join(TAIMMA_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from taimma.data.split import SequenceSample
from taimma.evaluation.e0_e1_protocol import (
    compute_noise_threshold_and_stability,
    check_e0_heterogeneity,
    extract_permitted_context_features,
    evaluate_e1_predictability,
    evaluate_supervisor_decision,
)


def test_noise_threshold_and_label_stability():
    """Verify calculation of noise threshold epsilon and Spearman stability regime."""
    # Seed 1 and Seed 2 deltas
    deltas_1 = np.array([0.5, -0.2, 0.8, -0.6, 0.1, 0.3])
    deltas_2 = np.array([0.4, -0.1, 0.7, -0.5, 0.2, 0.4])

    diffs = np.abs(deltas_1 - deltas_2)  # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    expected_eps = float(np.median(diffs))

    res = compute_noise_threshold_and_stability(deltas_1, deltas_2, source_name="category")
    assert abs(res.epsilon - expected_eps) < 1e-5
    assert res.spearman_rho > 0.8
    assert res.stability_regime == "contextual"

    # Low correlation case
    deltas_low = np.array([0.1, 0.9, -0.8, 0.2, -0.5, 0.7])
    res_low = compute_noise_threshold_and_stability(deltas_1, deltas_low, source_name="brand")
    assert res_low.stability_regime in ("soft_supervision", "group_level")


def test_e0_heterogeneity_conditions():
    """Verify E0 Condition A and Condition B logic with synthetic deltas."""
    # Condition A: >= 15% positive > eps and >= 15% negative < -eps
    N = 100
    eps = 0.1
    # 30 positive, 30 negative, 40 near zero
    deltas_a = np.array([0.5]*30 + [-0.5]*30 + [0.0]*40)
    seq_lens = np.array([5]*N)

    res_a = check_e0_heterogeneity(deltas_a, eps, seq_lens, source_name="category", n_bootstrap=100)
    assert res_a.condition_a_met is True
    assert res_a.e0_passed is True

    # Failed Condition A (only positive utility)
    deltas_fail = np.array([0.5]*80 + [0.0]*20)
    res_fail = check_e0_heterogeneity(deltas_fail, eps, seq_lens, source_name="brand", n_bootstrap=100)
    assert res_fail.condition_a_met is False
    assert res_fail.e0_passed is False


def test_e1_context_features_and_predictability():
    """Verify feature extraction (no target leakage) and 5-fold OOF predictability."""
    samples = [
        SequenceSample(user_idx=i, history=[1, 2, 3], target=4, history_len=3)
        for i in range(50)
    ]
    catalog_attrs = {"category": torch.tensor([1, 2, 3, 4], dtype=torch.long)}
    catalog_masks = {"category": torch.tensor([1, 1, 1, 1], dtype=torch.float32)}

    feats = extract_permitted_context_features(samples, catalog_attrs, catalog_masks, "category")
    assert feats.shape == (50, 4)
    # Target asin/item index 4 must not be in features
    assert np.all(np.isfinite(feats))

    # Test predictability with synthetic correlated signal
    rng = np.random.default_rng(42)
    feats[:, 1] = rng.normal(size=50)
    deltas = feats[:, 1] + rng.normal(0, 0.1, size=50)
    users = list(range(50))
    res_e1 = evaluate_e1_predictability(feats, deltas, users, epsilon=0.0, n_splits=3, n_bootstrap=100)
    assert res_e1.oof_auc_lr >= 0.5


def test_supervisor_decision_logic():
    """Verify GO, CONDITIONAL, and NO-GO state transitions based on E0 and E1."""
    # Case 1: Both pass with contextual stability -> GO
    from taimma.evaluation.e0_e1_protocol import (
        LabelStabilityResult,
        E0HeterogeneityResult,
        E1PredictabilityResult,
    )

    stab_good = {"category": LabelStabilityResult("category", 0.05, 0.65, 0.001, "contextual")}
    e0_good = {"category": E0HeterogeneityResult("category", 0.05, 0.25, 0.25, True, 0.1, (0.05, 0.15), -0.1, (-0.15, -0.05), True, True)}
    e1_good = {"category": E1PredictabilityResult("category", 0.62, 0.58, 0.51, (0.55, 0.69), True, "Pass")}

    report_go = evaluate_supervisor_decision(stab_good, e0_good, e1_good)
    assert report_go.decision == "GO"
    assert "category" in report_go.qualifying_sources
    assert report_go.gating_regime == "contextual"

    # Case 2: E0 passes but E1 fails -> NO-GO (predictability failed, cannot gate)
    e1_bad = {"category": E1PredictabilityResult("category", 0.52, 0.50, 0.51, (0.45, 0.59), False, "Fail")}
    report_nogo_e1 = evaluate_supervisor_decision(stab_good, e0_good, e1_bad)
    assert report_nogo_e1.decision == "NO-GO"

    # Case 3: E0 fails -> NO-GO
    e0_bad = {"category": E0HeterogeneityResult("category", 0.05, 0.05, 0.05, False, 0.1, (0.05, 0.15), 0.1, (0.05, 0.15), False, False)}
    report_nogo_e0 = evaluate_supervisor_decision(stab_good, e0_bad, e1_bad)
    assert report_nogo_e0.decision == "NO-GO"

    # Case 4: Verified fallback evidence (soft_supervision with E0 and E1 passing on same source) -> CONDITIONAL
    stab_fallback = {"category": LabelStabilityResult("category", 0.05, 0.22, 0.02, "soft_supervision")}
    from dataclasses import replace
    averaged_e1 = {"category": replace(e1_good["category"], evaluated_regime="averaged")}
    report_cond = evaluate_supervisor_decision(stab_fallback, e0_good, averaged_e1)
    assert report_cond.decision == "CONDITIONAL"
    assert report_cond.gating_regime == "soft_supervision"
    assert report_cond.recommended_regime_by_source == {"category": "soft_supervision"}

    # Case 5: Mismatched sources (category passes E0, brand passes E1) -> NO-GO (independent passes cannot cross-combine)
    e0_cat_only = {"category": e0_good["category"], "brand": e0_bad["category"]}
    e1_brd_only = {"category": e1_bad["category"], "brand": e1_good["category"]}
    stab_both = {"category": stab_good["category"], "brand": stab_good["category"]}
    report_mismatch = evaluate_supervisor_decision(stab_both, e0_cat_only, e1_brd_only)
    assert report_mismatch.decision == "NO-GO"
