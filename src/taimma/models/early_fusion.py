"""
early_fusion.py - SASRecF-style Early Fusion model for TAIMMA.

Architectural Distinction:
  - RecBole SASRecF uses feature concatenation: Linear([e_ID; e_cat; e_brand; e_title]).
  - TAIMMA Early Fusion specifies SASRecF-style EARLY ADDITION:
      x_t = e_ID(i_t) + SUM_{m in S} mask_m(i_t) * proj_m(e_m(i_t)) + position_embedding(t)

Supported Metadata Sources:
  - Category: multi-label / categorical attribute with embedding projection.
  - Brand: categorical attribute with 0 / missing handling.
  - Title: dense text representation with trainable linear projection.

Source Combinations:
  Supports all 8 subsets (7 non-empty combinations of {Category, Brand, Title} + ID-only).
"""

from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sasrec import SASRecModel


class EarlyFusionSASRec(nn.Module):
    """
    Early Fusion Sequential Recommender extending the base SASRec backbone.
    Adds projected metadata embeddings to ID embeddings before Transformer layers.
    """

    ALL_SOURCES = ("category", "brand", "title")

    def __init__(
        self,
        base_sasrec: SASRecModel,
        num_categories: int = 100,
        num_brands: int = 100,
        title_feature_dim: int = 64,
        sources_enabled: Tuple[str, ...] = ("category", "brand", "title"),
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.base_sasrec = base_sasrec
        self.num_items = base_sasrec.num_items
        self.max_seq_len = base_sasrec.max_seq_len
        self.embedding_dim = base_sasrec.embedding_dim
        self.sources_enabled = tuple(s.lower() for s in sources_enabled)

        # Source-specific projection / embedding layers
        # 1. Category embedding (padding / missing is 0)
        self.category_embedding = nn.Embedding(num_categories + 1, self.embedding_dim, padding_idx=0)
        # 2. Brand embedding (padding / missing is 0)
        self.brand_embedding = nn.Embedding(num_brands + 1, self.embedding_dim, padding_idx=0)
        # 3. Title projection from pre-extracted dense feature vector to embedding_dim
        self.title_projection = nn.Linear(title_feature_dim, self.embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.embedding_dim, eps=1e-12)

    def set_sources_enabled(self, sources: Tuple[str, ...]):
        """Dynamically configure active metadata source subset."""
        for s in sources:
            if s.lower() not in self.ALL_SOURCES:
                raise ValueError(f"Unknown source '{s}'. Allowed: {self.ALL_SOURCES}")
        self.sources_enabled = tuple(s.lower() for s in sources)

    def compute_source_contributions(
        self,
        category_ids: Optional[torch.Tensor] = None,
        brand_ids: Optional[torch.Tensor] = None,
        title_features: Optional[torch.Tensor] = None,
        category_mask: Optional[torch.Tensor] = None,
        brand_mask: Optional[torch.Tensor] = None,
        title_mask: Optional[torch.Tensor] = None,
        item_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Compute element-wise sum of active metadata source embeddings with availability masks.
        Handles sequence inputs (B, S), candidate catalog inputs (N,), and multilabel categories.
        """
        device = self.base_sasrec.item_embedding.weight.device

        # Determine item shape if not explicitly provided
        if item_shape is None:
            if brand_ids is not None:
                item_shape = brand_ids.shape
            elif title_features is not None:
                # Title features always has trailing feature dimension (D_title)
                item_shape = title_features.shape[:-1]
            elif category_ids is not None:
                if category_ids.dim() > 2:
                    # Multilabel sequence (B, S, L)
                    item_shape = category_ids.shape[:-1]
                else:
                    item_shape = category_ids.shape
            else:
                raise ValueError("At least one attribute tensor must be provided.")

        contributions = torch.zeros(*item_shape, self.embedding_dim, device=device)

        if "category" in self.sources_enabled and category_ids is not None:
            cat_embeds = self.category_embedding(category_ids)
            # If category_ids has an extra label dimension (multilabel: L > 1), pool across labels
            if category_ids.dim() > len(item_shape):
                non_zero = (category_ids > 0).unsqueeze(-1).float()
                sum_embeds = (cat_embeds * non_zero).sum(dim=-2)
                counts = non_zero.sum(dim=-2).clamp(min=1.0)
                cat_embeds = sum_embeds / counts
            if category_mask is not None:
                cat_embeds = cat_embeds * category_mask.unsqueeze(-1).float()
            contributions = contributions + cat_embeds

        if "brand" in self.sources_enabled and brand_ids is not None:
            brand_embeds = self.brand_embedding(brand_ids)
            if brand_mask is not None:
                brand_embeds = brand_embeds * brand_mask.unsqueeze(-1).float()
            contributions = contributions + brand_embeds

        if "title" in self.sources_enabled and title_features is not None:
            title_embeds = self.title_projection(title_features)
            if title_mask is not None:
                title_embeds = title_embeds * title_mask.unsqueeze(-1).float()
            contributions = contributions + title_embeds

        return contributions

    def encode_item_sequence(
        self,
        input_seqs: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        brand_ids: Optional[torch.Tensor] = None,
        title_features: Optional[torch.Tensor] = None,
        category_mask: Optional[torch.Tensor] = None,
        brand_mask: Optional[torch.Tensor] = None,
        title_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode user history sequence with early addition of metadata contributions:
          x_t = e_ID(i_t) + SUM_m e_m(i_t) + p(t)
        """
        B, S = input_seqs.size()
        device = input_seqs.device

        # Base ID embedding
        item_embeds = self.base_sasrec.item_embedding(input_seqs)

        # Early fusion addition if any source is enabled
        if len(self.sources_enabled) > 0:
            meta_contributions = self.compute_source_contributions(
                category_ids=category_ids,
                brand_ids=brand_ids,
                title_features=title_features,
                category_mask=category_mask,
                brand_mask=brand_mask,
                title_mask=title_mask,
                item_shape=(B, S),
            )
            item_embeds = item_embeds + meta_contributions

        # Add position embeddings
        positions = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.base_sasrec.position_embedding(positions)

        x = item_embeds + pos_embeds
        x = self.base_sasrec.input_layer_norm(x)
        x = self.dropout(x)

        padding_mask = (input_seqs == 0).unsqueeze(1).unsqueeze(2)
        causal_mask = self.base_sasrec.causal_mask[:S, :S]

        for block in self.base_sasrec.blocks:
            x = block(x, causal_mask=causal_mask, padding_mask=padding_mask)

        return self.base_sasrec.final_layer_norm(x)

    def encode_context(
        self,
        input_seqs: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        **meta_kwargs,
    ) -> torch.Tensor:
        """Extract user context representation at the last position."""
        hidden_states = self.encode_item_sequence(input_seqs, **meta_kwargs)
        if sequence_lengths is not None:
            idx = torch.full_like(sequence_lengths, hidden_states.size(1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embedding_dim)
            return hidden_states.gather(1, idx).squeeze(1)
        return hidden_states[:, -1, :]

    def score_candidates(
        self,
        context_embeddings: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        candidate_category_ids: Optional[torch.Tensor] = None,
        candidate_brand_ids: Optional[torch.Tensor] = None,
        candidate_title_features: Optional[torch.Tensor] = None,
        candidate_category_mask: Optional[torch.Tensor] = None,
        candidate_brand_mask: Optional[torch.Tensor] = None,
        candidate_title_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score candidate items whose representations are also early-fused:
          c_j = e_ID(j) + SUM_m e_m(j)
        """
        # ID embedding
        candidate_embeds = self.base_sasrec.item_embedding(candidate_item_ids)

        if len(self.sources_enabled) > 0:
            meta_contributions = self.compute_source_contributions(
                category_ids=candidate_category_ids,
                brand_ids=candidate_brand_ids,
                title_features=candidate_title_features,
                category_mask=candidate_category_mask,
                brand_mask=candidate_brand_mask,
                title_mask=candidate_title_mask,
                item_shape=candidate_item_ids.shape,
            )
            candidate_embeds = candidate_embeds + meta_contributions

        if candidate_item_ids.dim() == 1:
            return torch.matmul(context_embeddings, candidate_embeds.t())
        else:
            return torch.bmm(candidate_embeds, context_embeddings.unsqueeze(2)).squeeze(2)

    def score_full_catalog(
        self,
        context_embeddings: torch.Tensor,
        catalog_category_ids: Optional[torch.Tensor] = None,
        catalog_brand_ids: Optional[torch.Tensor] = None,
        catalog_title_features: Optional[torch.Tensor] = None,
        catalog_category_mask: Optional[torch.Tensor] = None,
        catalog_brand_mask: Optional[torch.Tensor] = None,
        catalog_title_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score all candidate catalog items (1..num_items)."""
        candidate_item_ids = torch.arange(
            1, self.num_items + 1, dtype=torch.long, device=context_embeddings.device
        )
        return self.score_candidates(
            context_embeddings,
            candidate_item_ids,
            candidate_category_ids=catalog_category_ids,
            candidate_brand_ids=catalog_brand_ids,
            candidate_title_features=catalog_title_features,
            candidate_category_mask=catalog_category_mask,
            candidate_brand_mask=catalog_brand_mask,
            candidate_title_mask=catalog_title_mask,
        )
