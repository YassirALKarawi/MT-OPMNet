"""Loss functions for multi-task OPM training (paper Section IV-D)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for MFI classification (Eq. 10 in paper).

    L_MFI = -Σ α_k (1 - p_k)^γ y_k ln(p_k)

    Args:
        gamma: Focusing parameter (γ_f = 2 in paper).
        reduction: 'mean', 'sum', or 'none'.
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
    """Homoscedastic uncertainty weighting (Eq. 7 in paper).

    L_total = exp(-s₁)·L_OSNR + exp(-s₂)·L_MFI + ½(s₁ + s₂)

    where s₁ = ln(σ₁²), s₂ = ln(σ₂²) are learnable scalars.

    Reference: Kendall et al., CVPR 2018.
    """

    def __init__(self):
        super().__init__()
        self.log_var_osnr = nn.Parameter(torch.zeros(1))
        self.log_var_mfi = nn.Parameter(torch.zeros(1))

    def forward(self, loss_osnr: torch.Tensor,
                loss_mfi: torch.Tensor) -> torch.Tensor:
        w_osnr = torch.exp(-self.log_var_osnr)
        w_mfi = torch.exp(-self.log_var_mfi)

        total = (w_osnr * loss_osnr + 0.5 * self.log_var_osnr
                 + w_mfi * loss_mfi + 0.5 * self.log_var_mfi)
        return total.squeeze()
