"""
sasrec.py - PyTorch implementation of SASRec (Self-Attentive Sequential Recommendation).

Reference:
  Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.

Features:
  - Causal multi-head self-attention with strict lower-triangular masking.
  - Left-padding awareness (padding mask for item index 0).
  - Tied item embedding table for candidate scoring.
  - Full-catalog Cross-Entropy (CE) loss as mandated by supervisor decisions.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal Multi-Head Self-Attention with pre-LayerNorm or post-LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float = 0.2):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, hidden_dim)
            causal_mask: Boolean tensor (seq_len, seq_len), True where attention is blocked
            padding_mask: Boolean tensor (batch_size, 1, 1, seq_len), True where key is padding
        """
        B, S, D = x.size()

        # Project and reshape: (B, num_heads, S, head_dim)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product: (B, num_heads, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            # Broadcast causal mask: (1, 1, S, S)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output context: (B, S, hidden_dim)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Standard Transformer Block: MHA + Pointwise FFN with LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float = 0.2, act: str = "gelu"):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.mha = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer_norm2 = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN MHA
        residual = x
        normed_x = self.layer_norm1(x)
        attn_out = self.mha(normed_x, causal_mask=causal_mask, padding_mask=padding_mask)
        x = residual + self.dropout1(attn_out)

        # Pre-LN FFN
        residual = x
        normed_x = self.layer_norm2(x)
        ffn_out = self.ffn(normed_x)
        x = residual + ffn_out
        return x


class SASRecModel(nn.Module):
    """
    SASRec ID-only sequential recommender.
    Provides sequence encoding, next-item full-catalog CE loss, and full ranking prediction.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        act: str = "gelu",
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Item embedding table: index 0 is padding (padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.input_layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, dropout_rate, act) for _ in range(num_blocks)]
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)

        # Cache causal upper-triangular mask (block future attention)
        # causal_mask[i, j] = True if j > i
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_item_sequence(
        self, input_seqs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode user history sequence into contextual representations for all positions.
        
        Args:
            input_seqs: LongTensor of shape (batch_size, max_seq_len)
            padding_mask: Optional boolean tensor of shape (batch_size, 1, 1, seq_len)
            
        Returns:
            hidden_states: Tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        B, S = input_seqs.size()
        device = input_seqs.device

        positions = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        item_embeds = self.item_embedding(input_seqs)
        pos_embeds = self.position_embedding(positions)

        x = item_embeds + pos_embeds
        x = self.input_layer_norm(x)
        x = self.dropout(x)

        if padding_mask is None:
            padding_mask = (input_seqs == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        causal_mask = self.causal_mask[:S, :S]

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, padding_mask=padding_mask)

        return self.final_layer_norm(x)

    def forward(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """Alias for encode_item_sequence."""
        return self.encode_item_sequence(input_seqs)

    def encode_context(
        self, input_seqs: torch.Tensor, sequence_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract user context embedding from the sequence.
        
        Args:
            input_seqs: LongTensor of shape (batch_size, max_seq_len)
            sequence_lengths: Optional LongTensor of shape (batch_size,) with valid length
            
        Returns:
            context_embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        hidden_states = self.encode_item_sequence(input_seqs)  # (B, S, D)
        if sequence_lengths is not None:
            # Gather at the last valid position: sequence_lengths - 1
            idx = torch.full_like(sequence_lengths, hidden_states.size(1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embedding_dim)
            return hidden_states.gather(1, idx).squeeze(1)
        return hidden_states[:, -1, :]  # Default to final position for left-padded sequence

    def get_user_representation(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for encode_context."""
        return self.encode_context(input_seqs)

    def score_candidates(
        self, context_embeddings: torch.Tensor, candidate_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scoring logits for specific candidate item IDs.
        
        Args:
            context_embeddings: Tensor of shape (batch_size, embedding_dim)
            candidate_item_ids: LongTensor of shape (batch_size, num_candidates) or (num_candidates,)
            
        Returns:
            candidate_scores: Tensor of shape (batch_size, num_candidates)
        """
        if candidate_item_ids.dim() == 1:
            # Broadcast across batch
            candidate_embeds = self.item_embedding(candidate_item_ids)  # (K, D)
            return torch.matmul(context_embeddings, candidate_embeds.t())  # (B, K)
        elif candidate_item_ids.dim() == 2:
            # Per-user candidate list: (B, K, D)
            candidate_embeds = self.item_embedding(candidate_item_ids)  # (B, K, D)
            return torch.bmm(candidate_embeds, context_embeddings.unsqueeze(2)).squeeze(2)  # (B, K)
        else:
            raise ValueError(f"candidate_item_ids must have 1 or 2 dimensions, got {candidate_item_ids.dim()}")

    def score_full_catalog(self, context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute logits across all catalog candidate items (1..num_items, excluding padding index 0).
        
        Args:
            context_embeddings: Tensor of shape (batch_size, embedding_dim)
            
        Returns:
            catalog_scores: Tensor of shape (batch_size, num_items)
        """
        candidate_embeds = self.item_embedding.weight[1 : self.num_items + 1]  # (num_items, D)
        return torch.matmul(context_embeddings, candidate_embeds.t())

    def predict_all(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """Compute full catalog logits from raw input sequences."""
        context = self.encode_context(input_seqs)
        return self.score_full_catalog(context)

    def compute_loss(self, input_seqs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute full-catalog Cross-Entropy loss.
        
        Args:
            input_seqs: LongTensor (B, S)
            targets: LongTensor (B,) with target indices in 1..num_items
            
        Returns:
            Scalar Cross-Entropy loss
        """
        logits = self.predict_all(input_seqs)  # (B, num_items)
        target_labels = targets - 1  # 0-based for cross_entropy
        return F.cross_entropy(logits, target_labels)

