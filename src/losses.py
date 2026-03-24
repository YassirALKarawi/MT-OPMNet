"""Loss functions for multi-task OPM training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.

    Applies a modulating factor (1 - p_t)^gamma to the standard cross-entropy
    loss, down-weighting easy examples and focusing on hard ones.

    Args:
        gamma: Focusing parameter. Higher values increase focus on hard samples.
        reduction: Reduction method ('mean', 'sum', or 'none').
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty weighting for multi-task learning.

    Learns log-variance parameters (one per task) that automatically
    balance the contribution of each task loss.

    Reference:
        Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
        Losses for Scene Geometry and Semantics", CVPR 2018.
    """

    def __init__(self):
        super().__init__()
        # Learnable log-variance for each task
        self.log_var_osnr = nn.Parameter(torch.zeros(1))
        self.log_var_mfi = nn.Parameter(torch.zeros(1))

    def forward(self, loss_osnr: torch.Tensor,
                loss_mfi: torch.Tensor) -> torch.Tensor:
        """Combine task losses with learned uncertainty weights.

        Args:
            loss_osnr: Scalar OSNR regression loss.
            loss_mfi: Scalar MFI classification loss.

        Returns:
            Weighted combined loss.
        """
        w_osnr = torch.exp(-self.log_var_osnr)
        w_mfi = torch.exp(-self.log_var_mfi)

        total = (w_osnr * loss_osnr + 0.5 * self.log_var_osnr
                 + w_mfi * loss_mfi + 0.5 * self.log_var_mfi)
        return total.squeeze()
