"""
Loss functions for imbalanced disaster tweet classification.

Why class imbalance matters in CrisisMMD:
  - The humanitarian task has 8 classes. "not_humanitarian" dominates at ~40%
    while "missing_or_found_people" is only ~2-3%.
  - A model that predicts "not_humanitarian" for everything achieves ~40%
    accuracy on that task — misleadingly high.
  - Macro F1 reveals this: it averages F1 per class equally, penalising
    the model heavily for ignoring minority classes.

Why accuracy alone is misleading:
  - Accuracy = (TP + TN) / N weights each sample equally, so majority-class
    samples dominate.  On a 90/10 binary task, always predicting the majority
    class yields 90% accuracy with zero recall on the minority class.
  - Macro F1 = mean(F1_c for c in classes) treats every class equally,
    regardless of support size.  This matches the real-world cost: failing to
    identify one injured person is just as bad as misclassifying ten.

Focal Loss (Lin et al., 2017):
  FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
  - (1 - p_t)^γ is the "modulating factor": easy examples (high p_t) get
    down-weighted; hard minority-class examples (low p_t) dominate the gradient.
  - γ=2 is the standard starting point.  Higher γ focuses even more on hard
    examples but can destabilise training on very small datasets.

Weighted CrossEntropy:
  - Simpler and more interpretable; assigns a static weight w_c to each class.
  - Works well when class imbalance is moderate.  Prefer Focal Loss when
    there is extreme imbalance (>10:1 ratio) or many confusable hard examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        weight:  per-class weight tensor (same role as in CrossEntropyLoss)
        gamma:   focusing parameter ≥ 0.  γ=0 reduces to weighted CE.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log_softmax is numerically stable
        log_prob = F.log_softmax(logits, dim=-1)
        prob     = log_prob.exp()

        # Gather p_t for each target class
        log_pt = log_prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt     = prob.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.weight is not None:
            w = self.weight.to(logits.device)
            alpha_t = w[targets]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Thin wrapper around nn.CrossEntropyLoss for consistency with FocalLoss API."""

    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits, targets, weight=w)


def build_loss(
    loss_type: str,
    class_weights: torch.Tensor,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory: return the configured loss module (weights kept on CPU; moved per batch)."""
    if loss_type == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma)
    return WeightedCrossEntropyLoss(weight=class_weights)
