"""
Model architectures for TAIMMA sequential recommenders.
"""

from .sasrec import SASRecModel
from .early_fusion import EarlyFusionSASRec
from .late_fusion import ResidualLateFusionModel, SourceSequentialEncoder
from .uasf import UASFModel, UASFContextGate, UASFGroupGate

__all__ = [
    "SASRecModel",
    "EarlyFusionSASRec",
    "ResidualLateFusionModel",
    "SourceSequentialEncoder",
    "UASFModel",
    "UASFContextGate",
    "UASFGroupGate",
]
