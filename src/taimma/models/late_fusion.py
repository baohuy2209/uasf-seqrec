"""
late_fusion.py - UASF-compatible Residual Late Fusion architecture for TAIMMA.

Architecture (Supervisor Decision October 20 & 26, 2026):
  - Stage S1a: Frozen ID-only SASRec foundation (s_ID).
  - Stage S1b: Jointly trained source-specific residual sequential encoders (s_m).
  - Combined score:
      s_joint(q, j) = s_ID(q, j) + SUM_{m in S} d_m * b_(j, m) * s_m(q, j)
    Where:
      - s_ID is frozen (no gradient updates).
      - d_m in {0, 1} is the source-dropout indicator.
      - b_(j, m) in {0, 1} is candidate-side metadata availability.
      - s_m is the source-specific residual score.
      - gamma_m = 1 (all-fixed mode).
  - Zero leakage of item-ID embeddings into source-only encoders.
"""

from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sasrec import MultiHeadAttention, SASRecModel, TransformerBlock


class SourceSequentialEncoder(nn.Module):
    """
    Lightweight sequential encoder for a single metadata source (Category, Brand, Title).
    Encodes exclusively source-specific historical sequences and candidate attributes.
    Zero dependency or leakage on item IDs.
    """

    def __init__(
        self,
        source_name: str,
        vocab_size: int,
        feature_dim: int = 64,
        max_seq_len: int = 50,
        num_blocks: int = 1,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        is_dense_feature: bool = False,
    ):
        super().__init__()
        self.source_name = source_name.lower()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.is_dense_feature = is_dense_feature

        if is_dense_feature:
            self.feature_projection = nn.Linear(vocab_size, feature_dim)
            self.candidate_projection = nn.Linear(vocab_size, feature_dim)
        else:
            # 0 is padding / unavailable
            self.source_embedding = nn.Embedding(vocab_size + 1, feature_dim, padding_idx=0)

        self.position_embedding = nn.Embedding(max_seq_len, feature_dim)
        self.input_layer_norm = nn.LayerNorm(feature_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [TransformerBlock(feature_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        )
        self.final_layer_norm = nn.LayerNorm(feature_dim, eps=1e-12)

        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def encode_source_sequence(self, source_seqs: torch.Tensor) -> torch.Tensor:
        """Encode metadata sequence: (B, S) -> (B, S, D)."""
        B, S = source_seqs.size()[:2]
        device = source_seqs.device

        if self.is_dense_feature:
            x = self.feature_projection(source_seqs)
            padding_mask = (source_seqs == 0).all(dim=-1).unsqueeze(1).unsqueeze(2)
        else:
            x = self.source_embedding(source_seqs)
            missing = source_seqs == 0
            if source_seqs.ndim == 3:
                valid = (~missing).unsqueeze(-1)
                x = (x * valid).sum(-2) / valid.sum(-2).clamp(min=1)
                missing = missing.all(-1)
            elif source_seqs.ndim != 2:
                raise ValueError("Categorical history must have shape [B,S] or [B,S,L]")
            padding_mask = missing.unsqueeze(1).unsqueeze(2)

        positions = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.position_embedding(positions)
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        causal_mask = self.causal_mask[:S, :S]
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, padding_mask=padding_mask)

        return self.final_layer_norm(x)

    def encode_context(self, source_seqs: torch.Tensor) -> torch.Tensor:
        """Extract user context representation for this source at final position: (B, D)."""
        hidden = self.encode_source_sequence(source_seqs)
        return hidden[:, -1, :]

    def get_candidate_embeddings(self, candidate_attrs: torch.Tensor) -> torch.Tensor:
        """Return candidate attribute embeddings: (num_candidates, D)."""
        if self.is_dense_feature:
            return self.candidate_projection(candidate_attrs)
        embedded = self.source_embedding(candidate_attrs)
        if candidate_attrs.ndim == 2:
            valid = (candidate_attrs > 0).unsqueeze(-1)
            return (embedded * valid).sum(-2) / valid.sum(-2).clamp(min=1)
        if candidate_attrs.ndim != 1:
            raise ValueError("Categorical candidates must have shape [N] or [N,L]")
        return embedded

    def score_candidates(
        self,
        context_embedding: torch.Tensor,
        candidate_attrs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute source residual score: s_m(q, j) = <h_m(q), c_m(j)>.
        Args:
            context_embedding: Tensor (B, D)
            candidate_attrs: Tensor (K,) or (K, feature_dim)
        Returns:
            scores: Tensor (B, K)
        """
        candidate_embeds = self.get_candidate_embeddings(candidate_attrs)  # (K, D)
        return torch.matmul(context_embedding, candidate_embeds.t())


class ResidualLateFusionModel(nn.Module):
    """
    Residual Late Fusion Model combining a frozen ID backbone with jointly trained
    source-specific residual encoders.
    """

    ALL_SOURCES = ("category", "brand", "title")

    def __init__(
        self,
        frozen_id_model: SASRecModel,
        source_encoders: Dict[str, SourceSequentialEncoder],
        p_drop: float = 0.2,
        sources_enabled: Tuple[str, ...] = ("category", "brand", "title"),
    ):
        super().__init__()
        self.id_model = frozen_id_model
        self.num_items = frozen_id_model.num_items
        self.p_drop = p_drop
        self.sources_enabled = tuple(s.lower() for s in sources_enabled)

        # Freeze the ID model parameters completely
        self.freeze_id_branch()

        # Register source encoders
        self.source_encoders = nn.ModuleDict(source_encoders)

    def freeze_id_branch(self):
        """Freeze ID backbone parameters so it acts as immutable reference."""
        for param in self.id_model.parameters():
            param.requires_grad = False
        self.id_model.eval()

    def set_sources_enabled(self, sources: Tuple[str, ...]):
        """Select active subset among the 8 possible source combinations."""
        for s in sources:
            if s.lower() not in self.ALL_SOURCES:
                raise ValueError(f"Unknown source '{s}'. Allowed: {self.ALL_SOURCES}")
        self.sources_enabled = tuple(s.lower() for s in sources)

    def compute_all_scores(
        self,
        item_id_seqs: torch.Tensor,
        source_seqs: Dict[str, torch.Tensor],
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
        apply_dropout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute frozen ID scores, individual residual source scores, and combined joint scores.

        Returns:
            scores_joint: (B, num_items)
            scores_id: (B, num_items) (detached / no grad)
            scores_per_source: Dict[source_name -> (B, num_items)]
        """
        # 1. Compute frozen ID scores (with torch.no_grad)
        with torch.no_grad():
            scores_id = self.id_model.predict_all(item_id_seqs)  # (B, num_items)

        scores_per_source: Dict[str, torch.Tensor] = {}
        residual_sum = torch.zeros_like(scores_id)

        # 2. Compute residual source scores
        for src_name, encoder in self.source_encoders.items():
            if src_name in self.sources_enabled:
                seq = source_seqs[src_name]
                catalog_attrs = catalog_source_attrs[src_name]
                avail_mask = catalog_availability_masks[src_name]  # (num_items,)

                ctx = encoder.encode_context(seq)  # (B, D)
                s_m = encoder.score_candidates(ctx, catalog_attrs)  # (B, num_items)
                scores_per_source[src_name] = s_m

                # Source dropout indicator d_m
                if self.training and apply_dropout and self.p_drop > 0.0:
                    d_m = float(torch.rand(1).item() > self.p_drop)
                else:
                    d_m = 1.0

                # Weighted by availability mask b_(j, m)
                masked_sm = s_m * avail_mask.unsqueeze(0).float()
                residual_sum = residual_sum + (d_m * masked_sm)

        scores_joint = scores_id + residual_sum
        return scores_joint, scores_id, scores_per_source

    def predict_all(
        self,
        item_id_seqs: torch.Tensor,
        source_seqs: Dict[str, torch.Tensor],
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict full-catalog ranking scores during evaluation."""
        scores_joint, _, _ = self.compute_all_scores(
            item_id_seqs=item_id_seqs,
            source_seqs=source_seqs,
            catalog_source_attrs=catalog_source_attrs,
            catalog_availability_masks=catalog_availability_masks,
            apply_dropout=False,
        )
        return scores_joint

    def compute_loss(
        self,
        item_id_seqs: torch.Tensor,
        source_seqs: Dict[str, torch.Tensor],
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Full-catalog Cross-Entropy loss for joint residual training."""
        scores_joint, _, _ = self.compute_all_scores(
            item_id_seqs=item_id_seqs,
            source_seqs=source_seqs,
            catalog_source_attrs=catalog_source_attrs,
            catalog_availability_masks=catalog_availability_masks,
            apply_dropout=True,
        )
        target_labels = targets - 1
        return F.cross_entropy(scores_joint, target_labels)
