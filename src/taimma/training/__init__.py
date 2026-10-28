"""
Training loop and model optimization for TAIMMA sequential recommenders.
"""

from .trainer import SASRecTrainer, TrainingHistory
from .multimodal_trainer import ResidualLateFusionTrainer, MultimodalSequenceDataset

__all__ = [
    "SASRecTrainer",
    "TrainingHistory",
    "ResidualLateFusionTrainer",
    "MultimodalSequenceDataset",
]
