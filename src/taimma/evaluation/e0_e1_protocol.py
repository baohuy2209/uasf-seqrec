"""
e0_e1_protocol.py - Utility Label Stability, E0 Heterogeneity, and E1 Predictability Protocol.

Strict adherence to Supervisor Decision Memo (October 26, 2026):
  - 10.1 Noise threshold: epsilon_m = median_q(|delta_m(q; seed_1) - delta_m(q; seed_2)|) on Train-meta.
  - 10.2 Label stability: Spearman correlation rho between 2 reference seeds.
  - 10.3 E0 Utility heterogeneity: Condition A (>=15% above +epsilon, >=15% below -epsilon)
         OR Condition B (Q1 and Q5 have opposite signs with 95% bootstrap CIs excluding zero).
  - 10.4 E1 Utility predictability: 5-fold user-grouped OOF CV with Logistic Regression & Tree,
         permitted context features only, 1000-sample bootstrap 95% CI.
  - 10.5 Decision policy: GO / CONDITIONAL / NO-GO.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import torch

from ..data.split import SequenceSample


# ---------------------------------------------------------------------------
# 10.1 & 10.2: Noise Threshold & Label Stability
# ---------------------------------------------------------------------------

@dataclass
class LabelStabilityResult:
    source_name: str
    epsilon: float
    spearman_rho: float
    p_value: float
    stability_regime: str  # contextual, soft_supervision, group_level
    spearman_ci: Optional[Tuple[float, float]] = None


def compute_noise_threshold_and_stability(
    deltas_seed1: np.ndarray,
    deltas_seed2: np.ndarray,
    source_name: str = "category",
    user_indices: Optional[List[int]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> LabelStabilityResult:
    """
    Compute epsilon_m and Spearman correlation between two reference seeds on Train-meta.
    """
    deltas_seed1 = np.asarray(deltas_seed1, dtype=float)
    deltas_seed2 = np.asarray(deltas_seed2, dtype=float)
    if deltas_seed1.ndim != 1 or len(deltas_seed1) == 0 or deltas_seed1.shape != deltas_seed2.shape:
        raise ValueError('Seed delta arrays must be nonempty, one-dimensional and aligned')
    if not np.isfinite(deltas_seed1).all() or not np.isfinite(deltas_seed2).all() or n_bootstrap < 1:
        raise ValueError('Reference deltas must be finite and bootstrap count positive')
    users = np.arange(len(deltas_seed1)) if user_indices is None else np.asarray(user_indices)
    if users.shape != deltas_seed1.shape:
        raise ValueError('User IDs must align with reference deltas')
    unique = np.unique(users)
    by_user = {u: np.flatnonzero(users == u) for u in unique}
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([by_user[u] for u in rng.choice(unique, len(unique), replace=True)])
        if np.ptp(deltas_seed1[idx]) > 0 and np.ptp(deltas_seed2[idx]) > 0:
            rhos.append(float(stats.spearmanr(deltas_seed1[idx], deltas_seed2[idx]).statistic))
    rho_ci = tuple(float(v) for v in np.percentile(rhos, [2.5, 97.5])) if rhos else None

    # 10.1 Noise threshold
    diffs = np.abs(deltas_seed1 - deltas_seed2)
    epsilon = float(np.median(diffs))

    # 10.2 Spearman rank correlation
    if np.all(deltas_seed1 == deltas_seed1[0]) or np.all(deltas_seed2 == deltas_seed2[0]):
        rho, p_val = 0.0, 1.0
    else:
        res = stats.spearmanr(deltas_seed1, deltas_seed2)
        rho, p_val = float(res.statistic), float(res.pvalue)

    if rho >= 0.3:
        regime = "contextual"
    elif rho >= 0.1:
        regime = "soft_supervision"
    else:
        regime = "group_level"

    return LabelStabilityResult(
        source_name=source_name,
        epsilon=epsilon,
        spearman_rho=rho,
        p_value=p_val,
        stability_regime=regime,
        spearman_ci=rho_ci,
    )


# ---------------------------------------------------------------------------
# 10.3: E0 Utility Heterogeneity
# ---------------------------------------------------------------------------

@dataclass
class E0HeterogeneityResult:
    source_name: str
    epsilon: float
    frac_positive: float
    frac_negative: float
    condition_a_met: bool
    q1_mean: float
    q1_ci: Tuple[float, float]
    q5_mean: float
    q5_ci: Tuple[float, float]
    condition_b_met: bool
    e0_passed: bool


def check_e0_heterogeneity(
    deltas: np.ndarray,
    epsilon: float,
    sequence_lengths: np.ndarray,
    source_name: str = "category",
    n_bootstrap: int = 1000,
    seed: int = 42,
    user_indices: Optional[List[int]] = None,
) -> E0HeterogeneityResult:
    """
    Evaluate Condition A and Condition B for E0 utility heterogeneity.
    """
    deltas = np.asarray(deltas, dtype=float)
    sequence_lengths = np.asarray(sequence_lengths, dtype=float)
    N = len(deltas)
    users = np.arange(N) if user_indices is None else np.asarray(user_indices)
    if N == 0 or deltas.shape != (N,) or sequence_lengths.shape != (N,) or users.shape != (N,):
        raise ValueError('E0 inputs must be nonempty and aligned')
    if not np.isfinite(deltas).all() or not np.isfinite(sequence_lengths).all() or not np.isfinite(epsilon) or epsilon < 0 or n_bootstrap < 1:
        raise ValueError('E0 inputs must be finite; epsilon nonnegative and bootstrap count positive')
    rng = np.random.default_rng(seed)

    # Condition A: >= 15% delta > epsilon AND >= 15% delta < -epsilon
    frac_pos = float(np.mean(deltas > epsilon))
    frac_neg = float(np.mean(deltas < -epsilon))
    cond_a = (frac_pos >= 0.15) and (frac_neg >= 0.15)

    # Condition B: Quintile Q1 (bottom 20%) and Q5 (top 20%) have opposite signs with bootstrap 95% CI excluding zero
    # Use rank-based disjoint partitioning to prevent quantile overlap and tie edge-effects
    n_samples = len(sequence_lengths)
    n_quintile = max(1, int(0.2 * n_samples))
    sorted_order = np.argsort(sequence_lengths)
    q1_idx = sorted_order[:n_quintile]
    q5_idx = sorted_order[-n_quintile:]
    # Ensure strictly disjoint
    q5_idx = np.setdiff1d(q5_idx, q1_idx)

    q1_deltas = deltas[q1_idx]
    q5_deltas = deltas[q5_idx] if len(q5_idx) > 0 else np.array([], dtype=deltas.dtype)

    def bootstrap_ci(indices):
        if len(indices) == 0:
            return None, (None, None)
        arr = deltas[indices]
        sample_users = users[indices]
        unique = np.unique(sample_users)
        by_user = {u: np.flatnonzero(sample_users == u) for u in unique}
        means = []
        for _ in range(n_bootstrap):
            idx = np.concatenate([by_user[u] for u in rng.choice(unique, len(unique), replace=True)])
            means.append(float(arr[idx].mean()))
        return float(arr.mean()), tuple(float(v) for v in np.percentile(means, [2.5, 97.5]))

    q1_mean, q1_ci = bootstrap_ci(q1_idx)
    q5_mean, q5_ci = bootstrap_ci(q5_idx)

    # Check opposite signs and neither CI covering 0
    if len(q1_deltas) >= 5 and len(q5_deltas) >= 5 and sequence_lengths[q1_idx].mean() < sequence_lengths[q5_idx].mean():
        opposite_signs = (q1_mean * q5_mean < 0.0)
        q1_excludes_zero = (q1_ci[0] > 0.0 or q1_ci[1] < 0.0)
        q5_excludes_zero = (q5_ci[0] > 0.0 or q5_ci[1] < 0.0)
        cond_b = opposite_signs and q1_excludes_zero and q5_excludes_zero
    else:
        cond_b = False

    e0_passed = cond_a or cond_b

    return E0HeterogeneityResult(
        source_name=source_name,
        epsilon=epsilon,
        frac_positive=round(frac_pos, 4),
        frac_negative=round(frac_neg, 4),
        condition_a_met=cond_a,
        q1_mean=q1_mean,
        q1_ci=q1_ci,
        q5_mean=q5_mean,
        q5_ci=q5_ci,
        condition_b_met=cond_b,
        e0_passed=e0_passed,
    )


# ---------------------------------------------------------------------------
# 10.4: E1 Utility Predictability (Permitted Context Features & OOF CV)
# ---------------------------------------------------------------------------

def extract_permitted_context_features(
    samples: List[SequenceSample],
    catalog_source_attrs: Dict[str, torch.Tensor],
    catalog_availability_masks: Dict[str, torch.Tensor],
    source_name: str,
) -> np.ndarray:
    """
    Extract legitimate inference-available context features psi_m(q):
      1. log(sequence_length)
      2. Source availability fraction in history
      3. Attribute unique count (diversity)
      4. Mean recency of source presence
    STRICTLY ZERO LEAKAGE: target identity or future interactions are not touched.
    """
    attrs = catalog_source_attrs[source_name].cpu().numpy()
    avail = catalog_availability_masks[source_name].cpu().numpy()

    features_list = []
    for s in samples:
        history = [itm for itm in s.history if itm > 0]
        L = max(len(history), 1)

        # 1. log sequence length
        log_len = np.log(L)

        # 2. Historical source coverage
        has_src = [
            1.0 if (itm <= len(avail) and avail[itm - 1] > 0) else 0.0
            for itm in history
        ]
        src_cov = np.mean(has_src) if has_src else 0.0

        # Shannon entropy of observed categorical labels (including multilabel).
        values = []
        for itm in history:
            if itm > len(attrs) or avail[itm - 1] <= 0:
                continue
            value = attrs[itm - 1]
            if attrs.ndim == 1:
                if value > 0:
                    values.append(int(value))
            elif np.issubdtype(attrs.dtype, np.integer):
                values.extend(int(v) for v in value if v > 0)
            else:
                values.append(tuple(value.tolist()))
        from collections import Counter
        counts = np.asarray(list(Counter(values).values()), dtype=float)
        probs = counts / counts.sum() if len(counts) else counts
        attr_entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0

        # 4. Recent source indicator (last 3 items)
        recent_has_src = np.mean(has_src[-3:]) if len(has_src) >= 3 else src_cov

        features_list.append([log_len, src_cov, attr_entropy, recent_has_src])

    return np.array(features_list, dtype=np.float32)


@dataclass
class E1PredictabilityResult:
    source_name: str
    oof_auc_lr: float
    oof_auc_tree: float
    baseline_len_auc: float
    bootstrap_95_ci: Tuple[float, float]
    e1_passed: bool
    evidence_summary: str
    oof_spearman_rho: Optional[float] = None
    oof_spearman_ci: Optional[Tuple[float, float]] = None
    status: str = 'EVALUATED'
    evaluated_regime: str = 'contextual'


def evaluate_e1_predictability(
    context_features: np.ndarray,
    utility_deltas: np.ndarray,
    user_indices: List[int],
    epsilon: float,
    source_name: str = 'category',
    n_splits: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    evaluated_regime: str = 'contextual',
    group_indices: Optional[np.ndarray] = None,
) -> E1PredictabilityResult:
    """User-held-out contextual or group-lookup prediction, with clustered CIs.

    Features column zero is log(length). Group IDs are supplied from historical
    information only: 2 * length_quintile + history_head_tail (0..9). Group
    prediction uses training-fold label means; held-out labels never fit gates.
    Undefined metrics are returned as None with NOT_EVALUABLE, never as 0.5.
    """
    X = np.asarray(context_features, dtype=float)
    delta = np.asarray(utility_deltas, dtype=float)
    groups = np.asarray(user_indices)
    n = len(delta)
    if X.ndim != 2 or X.shape[0] != n or X.shape[1] < 1 or groups.shape != (n,):
        raise ValueError('Misaligned E1 features, utility labels or user IDs')
    if n == 0 or not np.isfinite(X).all() or not np.isfinite(delta).all() or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('E1 inputs must be nonempty and finite; epsilon must be nonnegative')
    if n_bootstrap < 1 or n_splits < 2:
        raise ValueError('Require positive bootstrap count and at least two CV folds')
    if evaluated_regime not in ('contextual', 'averaged', 'group'):
        raise ValueError('Unknown E1 evaluation regime')

    def undefined(reason):
        return E1PredictabilityResult(source_name, None, None, None, None, False, reason,
            oof_spearman_rho=None, status='NOT_EVALUABLE', evaluated_regime=evaluated_regime)

    y = (delta > epsilon).astype(int)
    unique_users = np.unique(groups)
    if len(np.unique(y)) < 2:
        return undefined('Utility labels contain only one class; AUC is undefined')
    if len(unique_users) < n_splits:
        return undefined(f'Need at least {n_splits} distinct users for the configured grouped CV')
    gate_groups = None
    if evaluated_regime == 'group':
        if group_indices is None:
            return undefined('Group fallback requires explicit inference-available group_idx; not computed')
        gate_groups = np.asarray(group_indices)
        if gate_groups.shape != (n,) or not np.isin(gate_groups, np.arange(10)).all():
            raise ValueError('group_idx must be aligned integer IDs 0..9')

    predictions = np.empty(n)
    tree_predictions = np.empty(n)
    length_predictions = np.empty(n)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            return undefined('A training fold has only one class; grouped OOF cannot be fitted')
        scaler = StandardScaler().fit(X[train_idx])
        train_X, test_X = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        if evaluated_regime == 'group':
            train_groups, test_groups = gate_groups[train_idx], gate_groups[test_idx]
            means = {g: float(y[train_idx][train_groups == g].mean()) for g in np.unique(train_groups)}
            if any(g not in means for g in np.unique(test_groups)):
                return undefined('Held-out group absent from training fold; no group-gate evidence')
            predictions[test_idx] = [means[g] for g in test_groups]
            tree_predictions[test_idx] = predictions[test_idx]
        else:
            lr = LogisticRegression(random_state=seed, max_iter=200).fit(train_X, y[train_idx])
            tree = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(train_X, y[train_idx])
            predictions[test_idx] = lr.predict_proba(test_X)[:, 1]
            tree_predictions[test_idx] = tree.predict_proba(test_X)[:, 1]
        length_lr = LogisticRegression(random_state=seed, max_iter=200).fit(train_X[:, :1], y[train_idx])
        length_predictions[test_idx] = length_lr.predict_proba(test_X[:, :1])[:, 1]

    auc = float(roc_auc_score(y, predictions))
    auc_tree = float(roc_auc_score(y, tree_predictions))
    auc_length = float(roc_auc_score(y, length_predictions))
    def spearman(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(stats.spearmanr(a, b).statistic)
    rho = spearman(predictions, delta)
    rng = np.random.default_rng(seed)
    indices_by_user = {u: np.flatnonzero(groups == u) for u in unique_users}
    bootstrap_aucs, bootstrap_rhos = [], []
    for _ in range(n_bootstrap):
        idx = np.concatenate([indices_by_user[u] for u in rng.choice(unique_users, len(unique_users), replace=True)])
        if len(np.unique(y[idx])) > 1:
            bootstrap_aucs.append(roc_auc_score(y[idx], predictions[idx]))
        value = spearman(predictions[idx], delta[idx])
        if value is not None:
            bootstrap_rhos.append(value)
    if not bootstrap_aucs:
        return undefined('No bootstrap replicate had a defined AUC')
    ci = tuple(float(v) for v in np.percentile(bootstrap_aucs, [2.5, 97.5]))
    rho_ci = tuple(float(v) for v in np.percentile(bootstrap_rhos, [2.5, 97.5])) if bootstrap_rhos else None
    passed = bool(auc >= 0.55 and ci[0] > 0.5 and auc > auc_length)
    return E1PredictabilityResult(source_name, auc, auc_tree, auc_length, ci, passed,
        f'{evaluated_regime} user-grouped OOF; AUC={auc:.6f}; length={auc_length:.6f}; valid AUC bootstrap={len(bootstrap_aucs)}/{n_bootstrap}',
        oof_spearman_rho=rho, oof_spearman_ci=rho_ci, evaluated_regime=evaluated_regime)


def make_utility_labels(deltas, epsilon: float, temperature: Optional[float] = None):
    """Temperature must be explicitly supplied for the prescribed soft-label regime."""
    values = np.asarray(deltas, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Utility labels require finite deltas and nonnegative epsilon')
    if temperature is None:
        return (values > epsilon).astype(float)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Soft-label temperature must be explicitly positive')
    from scipy.special import expit
    return expit((values - epsilon) / temperature)


# ---------------------------------------------------------------------------
# 10.5: Supervisor Decision Engine (GO / CONDITIONAL / NO-GO)
# ---------------------------------------------------------------------------

@dataclass
class SupervisorDecisionReport:
    decision: str  # GO, CONDITIONAL, NO-GO, or PENDING for missing/undefined evidence
    rationale: str
    per_source_e0: Dict[str, E0HeterogeneityResult]
    per_source_e1: Dict[str, E1PredictabilityResult]
    stability_results: Dict[str, LabelStabilityResult]
    qualifying_sources: List[str] = field(default_factory=list)
    gating_regime: str = "none"


def evaluate_supervisor_decision(
    stability_results: Dict[str, LabelStabilityResult],
    e0_results: Dict[str, E0HeterogeneityResult],
    e1_results: Dict[str, E1PredictabilityResult],
) -> SupervisorDecisionReport:
    """
    Apply locked October 26 decision policy to determine GO / CONDITIONAL / NO-GO.

    Policy Invariants:
      - GO requires that AT LEAST ONE source simultaneously satisfies E0 heterogeneity,
        E1 predictability, and contextual label stability (rho >= 0.3) ON THAT SAME SOURCE.
      - CONDITIONAL requires that at least one source satisfies E0 heterogeneity and E1 predictability,
        but its stability regime requires soft supervision (0.1 <= rho < 0.3) or group-level gating (rho < 0.1).
      - NO-GO: If across all sources, either E0 or E1 fails. E0 passing while E1 fails
        must NEVER automatically produce CONDITIONAL without verified fallback predictability.
    """
    go_sources: List[str] = []
    conditional_sources: List[Tuple[str, str]] = []
    source_reasons: List[str] = []
    pending = not e0_results
    if set(e0_results) != set(e1_results) or set(e0_results) != set(stability_results):
        pending = True

    for src in e0_results:
        e0 = e0_results[src]
        e1 = e1_results.get(src)
        stab = stability_results.get(src)

        if e1 is None or stab is None:
            source_reasons.append(f"{src}: Missing E1 or stability result")
            pending = True
            continue
        if e1.status != 'EVALUATED':
            pending = True
            source_reasons.append(f'{src}: {e1.evidence_summary}')
            continue

        if e0.e0_passed and e1.e1_passed:
            if stab.stability_regime == "contextual" and e1.evaluated_regime == 'contextual':
                go_sources.append(src)
            elif ((stab.stability_regime == 'soft_supervision' and e1.evaluated_regime == 'averaged')
                  or (stab.stability_regime == 'group_level' and e1.evaluated_regime == 'group')):
                conditional_sources.append((src, stab.stability_regime))
            else:
                pending = True
                source_reasons.append(f'{src}: required fallback predictability was not evaluated')
        else:
            source_reasons.append(
                f"{src}: E0={e0.e0_passed}, E1={e1.e1_passed}, stability={stab.stability_regime}"
            )

    if len(go_sources) > 0:
        decision = "GO"
        gating_regime = "contextual"
        qualifying = go_sources
        rationale = (
            f"GO: Source(s) {go_sources} satisfied both E0 (utility heterogeneity) and E1 (predictability) "
            f"with reliable contextual label stability (rho >= 0.3). Authorized to proceed to full S3 UASF contextual gating."
        )
    elif len(conditional_sources) > 0:
        decision = "CONDITIONAL"
        qualifying = [src for src, _ in conditional_sources]
        gating_regime = "group_level"
        cond_details = [f"{src} ({regime})" for src, regime in conditional_sources]
        rationale = (
            f"CONDITIONAL: Source(s) {cond_details} satisfied E0 and E1, but stability regime requires "
            f"prescribed label averaging / soft supervision (0.1 <= rho < 0.3) or group-level gating (rho < 0.1). "
            f"Deploy UASF as group-level gating per Memo 26/10."
        )
    else:
        decision = "PENDING" if pending else "NO-GO"
        gating_regime = "none"
        qualifying = []
        rationale = (
            f"{decision}: No source has sufficient validated E0/E1 evidence. "
            f"Details: {'; '.join(source_reasons) if source_reasons else 'No sources provided'}. "
            f"Do not infer uniform utility or absence of context dependence from missing or failed predictability evidence."
        )

    return SupervisorDecisionReport(
        decision=decision,
        rationale=rationale,
        per_source_e0=e0_results,
        per_source_e1=e1_results,
        stability_results=stability_results,
        qualifying_sources=qualifying,
        gating_regime=gating_regime,
    )
