"""
Experiment manifest tracking and result registry for TAIMMA.
"""

from .logger import ExperimentLogger
from .diagnostics import (
    compute_marginal_utility_and_redundancy,
    permute_metadata_train_time,
    permute_metadata_test_time,
    generate_categorical_random_control,
    generate_title_gaussian_control,
    generate_item_missingness_mask,
    partition_by_sequence_length_quintiles,
    partition_by_item_popularity,
    compute_negative_transfer_rate,
    MarginalUtilityResult,
)

__all__ = [
    "ExperimentLogger",
    "compute_marginal_utility_and_redundancy",
    "permute_metadata_train_time",
    "permute_metadata_test_time",
    "generate_categorical_random_control",
    "generate_title_gaussian_control",
    "generate_item_missingness_mask",
    "partition_by_sequence_length_quintiles",
    "partition_by_item_popularity",
    "compute_negative_transfer_rate",
    "MarginalUtilityResult",
]
