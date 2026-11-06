"""One context-level source mask, shared by history and candidate scoring."""
from typing import Sequence
import torch

SOURCES = ("category", "brand", "title")


def validate_source_weights(weights, batch_size):
    if weights.ndim != 2 or tuple(weights.shape) != (batch_size, len(SOURCES)):
        raise ValueError("source_weights must have shape [batch, 3] in category/brand/title order")
    if not torch.isfinite(weights).all() or torch.any(weights < 0) or torch.any(weights > 1):
        raise ValueError("Source weights must be finite and between zero and one")


def sample_source_weights(batch_size: int, enabled: Sequence[str], device,
                          p_drop: float = 0.0, p_all_off: float = 0.0,
                          training: bool = False, dtype=torch.float32):
    if not 0 <= p_drop <= 1 or not 0 <= p_all_off <= 1:
        raise ValueError("Drop probabilities must be in [0, 1]")
    if any(s not in SOURCES for s in enabled):
        raise ValueError("Unknown metadata source")
    weights = torch.ones(batch_size, len(SOURCES), device=device, dtype=dtype)
    if training and p_drop:
        weights = (torch.rand(batch_size, len(SOURCES), device=device) >= p_drop).to(dtype)
    if training and p_all_off:
        weights = weights * (torch.rand(batch_size, 1, device=device) >= p_all_off).to(dtype)
    active = torch.tensor([s in enabled for s in SOURCES], device=device, dtype=dtype)
    return weights * active.unsqueeze(0)
