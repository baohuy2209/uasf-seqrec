"""
Independent evaluation and baseline models for TAIMMA sequential recommenders.
"""

from .evaluator import IndependentEvaluator, EvaluationResult, UserRankRecord
from .baselines import RandomBaseline, PopularityBaseline
from .utility import (
    compute_leave_one_source_out_deltas,
    export_utility_records_to_df,
    UtilityRecord,
)
from .e0_e1_protocol import (
    compute_noise_threshold_and_stability,
    check_e0_heterogeneity,
    extract_permitted_context_features,
    evaluate_e1_predictability,
    evaluate_supervisor_decision,
    LabelStabilityResult,
    E0HeterogeneityResult,
    E1PredictabilityResult,
    SupervisorDecisionReport,
)
from .statistics import (
    paired_bootstrap_test,
    holm_bonferroni_correction,
    tost_equivalence_test,
    BootstrapComparisonResult,
    FamilyComparisonReport,
)

__all__ = [
    "IndependentEvaluator",
    "EvaluationResult",
    "UserRankRecord",
    "RandomBaseline",
    "PopularityBaseline",
    "compute_leave_one_source_out_deltas",
    "export_utility_records_to_df",
    "UtilityRecord",
    "compute_noise_threshold_and_stability",
    "check_e0_heterogeneity",
    "extract_permitted_context_features",
    "evaluate_e1_predictability",
    "evaluate_supervisor_decision",
    "LabelStabilityResult",
    "E0HeterogeneityResult",
    "E1PredictabilityResult",
    "SupervisorDecisionReport",
    "paired_bootstrap_test",
    "holm_bonferroni_correction",
    "tost_equivalence_test",
    "BootstrapComparisonResult",
    "FamilyComparisonReport",
]
