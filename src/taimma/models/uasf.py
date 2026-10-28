"""
uasf.py - Utility-Supervised Selective Fusion (UASF) Model and Comparative Baselines.

Architecture & Formulation (Supervisor Decision October 20 & 26, 2026):
  Contextual Gate:
    beta_m(q) = sigmoid(g_phi(psi_m(q)) / tau)
    alpha_m(q) = a_m(q) * beta_m(q)
    At inference: if beta_m(q) < kappa, alpha_m(q) = 0.
  
  Prediction Function:
    score(q, i) = s_ID(q, i) + SUM_{m in S} b_(i, m) * alpha_m(q) * s_m(q, i)
    Gamma is fixed to 1.
    Exact ID-only score recovery when all alpha_m(q) = 0.
  
  Joint Objective:
    L_total = L_rec + lambda_util * L_util + lambda_sparse * L_sparse
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import SASRecModel
from .late_fusion import ResidualLateFusionModel, SourceSequentialEncoder


class UASFContextGate(nn.Module):
    """
    Contextual Gating Network g_phi(psi(q)).
    Maps Permitted Context Features psi(q) -> Gate Activation beta_m(q).
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int = 3,
        hidden_dim: int = 32,
        tau: float = 1.0,
        kappa: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.tau = tau
        self.kappa = kappa

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sources),
        )

    def forward(
        self,
        context_features: torch.Tensor,
        history_availability: torch.Tensor,
        apply_threshold: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_features: Tensor (B, input_dim)
            history_availability: Tensor (B, num_sources), a_m(q) in {0, 1}
            apply_threshold: If True (inference), zero out gates below kappa.

        Returns:
            alpha: Gated source weights (B, num_sources)
            beta: Raw pre-threshold probabilities (B, num_sources)
        """
        logits = self.mlp(context_features)  # (B, num_sources)
        beta = torch.sigmoid(logits / self.tau)

        # History availability masking
        alpha = beta * history_availability.float()

        if apply_threshold and self.kappa > 0.0:
            # If beta < kappa, source is disabled (alpha = 0)
            threshold_mask = (beta >= self.kappa).float()
            alpha = alpha * threshold_mask

        return alpha, beta


class UASFGroupGate(nn.Module):
    """
    Group-level gate fallback (when label stability rho < 0.1).
    Maps discrete group index -> static gate activation.
    """

    def __init__(self, num_groups: int = 10, num_sources: int = 3):
        super().__init__()
        self.num_groups = num_groups
        self.num_sources = num_sources
        self.group_weights = nn.Parameter(torch.zeros(num_groups, num_sources))

    def forward(
        self,
        group_indices: torch.Tensor,
        history_availability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.group_weights[group_indices]
        beta = torch.sigmoid(logits)
        alpha = beta * history_availability.float()
        return alpha, beta


class UASFModel(nn.Module):
    """
    Utility-Supervised Selective Fusion (UASF) Model.
    Combines frozen ID backbone, frozen residual source encoders, and trainable selective gate.
    """

    SOURCES = ("category", "brand", "title")

    def __init__(
        self,
        frozen_late_fusion_model: ResidualLateFusionModel,
        context_feature_dim: int = 4,
        hidden_dim: int = 32,
        tau: float = 1.0,
        kappa: float = 0.1,
        lambda_util: float = 0.5,
        lambda_sparse: float = 0.001,
        gate_mode: str = "contextual",  # 'contextual', 'group', 'all_fixed', 'id_only', 'learned_global', 'softmax'
        num_groups: int = 10,
    ):
        super().__init__()
        self.late_fusion = frozen_late_fusion_model
        self.num_items = frozen_late_fusion_model.num_items
        self.sources = list(frozen_late_fusion_model.sources_enabled)
        self.num_sources = len(self.sources)
        self.tau = tau
        self.kappa = kappa
        self.lambda_util = lambda_util
        self.lambda_sparse = lambda_sparse
        self.gate_mode = gate_mode

        # Freeze all backbone and residual parameters
        for param in self.late_fusion.parameters():
            param.requires_grad = False
        self.late_fusion.eval()

        # Gate configurations
        if gate_mode == "contextual":
            self.gate = UASFContextGate(
                input_dim=context_feature_dim,
                num_sources=self.num_sources,
                hidden_dim=hidden_dim,
                tau=tau,
                kappa=kappa,
            )
        elif gate_mode == "group":
            self.gate = UASFGroupGate(num_groups=num_groups, num_sources=self.num_sources)
        elif gate_mode == "learned_global":
            self.global_weights = nn.Parameter(torch.zeros(self.num_sources))
        elif gate_mode == "softmax":
            # Softmax gate includes ID/null option at index 0
            self.softmax_mlp = nn.Sequential(
                nn.Linear(context_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_sources + 1),
            )

    def train(self, mode: bool = True):
        """
        Set training mode for trainable gate parameters while keeping frozen
        late fusion reference backbone and residual branches strictly in eval mode (disabling dropout).
        """
        super().train(mode)
        if hasattr(self, "late_fusion") and self.late_fusion is not None:
            self.late_fusion.eval()
        return self

    def compute_gate_weights(
        self,
        context_features: Optional[torch.Tensor] = None,
        history_availability: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
        apply_threshold: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alpha and beta weights based on active gate mode.
        """
        B = context_features.size(0) if context_features is not None else group_indices.size(0)
        device = self.late_fusion.id_model.item_embedding.weight.device

        if history_availability is None:
            history_availability = torch.ones(B, self.num_sources, device=device)

        if self.gate_mode == "id_only":
            alpha = torch.zeros(B, self.num_sources, device=device)
            beta = torch.zeros(B, self.num_sources, device=device)
            return alpha, beta

        if self.gate_mode == "all_fixed":
            beta = torch.ones(B, self.num_sources, device=device)
            alpha = beta * history_availability.float()
            return alpha, beta

        if self.gate_mode == "learned_global":
            beta = torch.sigmoid(self.global_weights).unsqueeze(0).expand(B, -1)
            alpha = beta * history_availability.float()
            return alpha, beta

        if self.gate_mode == "softmax":
            logits = self.softmax_mlp(context_features)  # (B, num_sources + 1)
            probs = F.softmax(logits, dim=-1)
            # Source gates are indices 1..num_sources
            beta = probs[:, 1:]
            alpha = beta * history_availability.float()
            return alpha, beta

        if self.gate_mode == "group":
            return self.gate(group_indices, history_availability)

        # Default: Contextual gate
        return self.gate(context_features, history_availability, apply_threshold=apply_threshold)

    def forward_scores(
        self,
        item_id_seqs: torch.Tensor,
        source_seqs: Dict[str, torch.Tensor],
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
        context_features: Optional[torch.Tensor] = None,
        history_availability: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
        apply_threshold: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute full catalog scores under UASF:
          score(q, i) = s_ID(q, i) + sum_m b_(i, m) * alpha_m(q) * s_m(q, i)
        
        Returns:
            scores_uasf: Tensor (B, num_items)
            alpha: Gated active weights (B, num_sources)
            beta: Raw gate activations (B, num_sources)
        """
        # 1. Compute frozen ID scores and source residual scores
        with torch.no_grad():
            scores_joint, scores_id, scores_per_source = self.late_fusion.compute_all_scores(
                item_id_seqs=item_id_seqs,
                source_seqs=source_seqs,
                catalog_source_attrs=catalog_source_attrs,
                catalog_availability_masks=catalog_availability_masks,
                apply_dropout=False,
            )

        # 2. Compute gate weights
        alpha, beta = self.compute_gate_weights(
            context_features=context_features,
            history_availability=history_availability,
            group_indices=group_indices,
            apply_threshold=apply_threshold,
        )

        # 3. Fuse scores with gamma = 1
        scores_uasf = scores_id.clone()
        for idx, src_name in enumerate(self.sources):
            if src_name in scores_per_source:
                s_m = scores_per_source[src_name]
                b_m = catalog_availability_masks[src_name].unsqueeze(0).float()  # (1, num_items)
                alpha_m = alpha[:, idx].unsqueeze(1)  # (B, 1)

                scores_uasf = scores_uasf + (b_m * alpha_m * s_m)

        return scores_uasf, alpha, beta

    def predict_all(
        self,
        item_id_seqs: torch.Tensor,
        source_seqs: Dict[str, torch.Tensor],
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
        context_features: Optional[torch.Tensor] = None,
        history_availability: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference scoring with thresholding enabled."""
        scores_uasf, _, _ = self.forward_scores(
            item_id_seqs=item_id_seqs,
            source_seqs=source_seqs,
            catalog_source_attrs=catalog_source_attrs,
            catalog_availability_masks=catalog_availability_masks,
            context_features=context_features,
            history_availability=history_availability,
            group_indices=group_indices,
            apply_threshold=True,
        )
        return scores_uasf

    def compute_loss(
        self,
        item_id_seqs: torch.Tensor,
        source_seqs: Dict[str, torch.Tensor],
        catalog_source_attrs: Dict[str, torch.Tensor],
        catalog_availability_masks: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        utility_labels: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        history_availability: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute UASF total objective:
          L_total = L_rec + lambda_util * L_util + lambda_sparse * L_sparse
        """
        scores, alpha, beta = self.forward_scores(
            item_id_seqs=item_id_seqs,
            source_seqs=source_seqs,
            catalog_source_attrs=catalog_source_attrs,
            catalog_availability_masks=catalog_availability_masks,
            context_features=context_features,
            history_availability=history_availability,
            group_indices=group_indices,
            apply_threshold=False,
        )

        target_labels = targets - 1
        l_rec = F.cross_entropy(scores, target_labels)

        # Utility supervision loss (BCE against detached utility labels)
        if utility_labels is not None and self.lambda_util > 0.0:
            l_util = F.binary_cross_entropy(beta, utility_labels.float())
        else:
            l_util = torch.tensor(0.0, device=scores.device)

        # Sparsity penalty: penalizes source activation
        if self.lambda_sparse > 0.0:
            l_sparse = torch.mean(beta)
        else:
            l_sparse = torch.tensor(0.0, device=scores.device)

        total_loss = l_rec + (self.lambda_util * l_util) + (self.lambda_sparse * l_sparse)

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_rec": float(l_rec.item()),
            "loss_util": float(l_util.item()),
            "loss_sparse": float(l_sparse.item()),
        }
        return total_loss, loss_dict
