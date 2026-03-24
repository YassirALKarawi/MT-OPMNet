"""
MT-OPMNet model architectures.

Includes:
  - CAAM: Channel-Aware Attention Module
  - MTOPMNet: Multi-task model (OSNR + MFI) with optional CAAM
  - SingleTaskOSNR: Single-task baseline for OSNR estimation
  - SingleTaskMFI: Single-task baseline for modulation format identification
  - FocalLoss: Focal loss for class-imbalanced MFI
  - UncertaintyWeighting: Learnable homoscedastic uncertainty task weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel-Aware Attention Module (CAAM)
# ---------------------------------------------------------------------------

class CAAM(nn.Module):
    """Squeeze-and-Excitation style channel attention for 1-D feature maps.

    GAP -> FC -> Sigmoid -> channel reweighting.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        w = x.mean(dim=2)          # GAP -> (B, C)
        w = self.fc(w).unsqueeze(2) # (B, C, 1)
        return x * w


# ---------------------------------------------------------------------------
# Shared Conv1D backbone
# ---------------------------------------------------------------------------

class ConvBackbone(nn.Module):
    """Four-layer Conv1D feature extractor."""

    def __init__(self, use_caam=True):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in [32, 64, 128, 256]:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.caam = CAAM(256) if use_caam else None

    def forward(self, x):
        # x: (B, 1, 100)
        x = self.conv(x)           # (B, 256, 6)  [100 / 2^4 = 6]
        if self.caam is not None:
            x = self.caam(x)
        x = x.mean(dim=2)          # GAP -> (B, 256)
        return x


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class OSNRHead(nn.Module):
    """Regression head: shared_repr -> OSNR estimate (scalar)."""

    def __init__(self, in_features=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MFIHead(nn.Module):
    """Classification head: shared_repr -> modulation format logits."""

    def __init__(self, in_features=256, n_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------

class MTOPMNet(nn.Module):
    """Multi-task model: joint OSNR estimation + MFI."""

    def __init__(self, n_classes=5, use_caam=True):
        super().__init__()
        self.backbone = ConvBackbone(use_caam=use_caam)
        self.osnr_head = OSNRHead(256)
        self.mfi_head = MFIHead(256, n_classes)

    def forward(self, x):
        feat = self.backbone(x)
        osnr = self.osnr_head(feat)
        mfi_logits = self.mfi_head(feat)
        return osnr, mfi_logits


class SingleTaskOSNR(nn.Module):
    """Single-task baseline for OSNR regression."""

    def __init__(self, use_caam=True):
        super().__init__()
        self.backbone = ConvBackbone(use_caam=use_caam)
        self.head = OSNRHead(256)

    def forward(self, x):
        return self.head(self.backbone(x))


class SingleTaskMFI(nn.Module):
    """Single-task baseline for modulation format identification."""

    def __init__(self, n_classes=5, use_caam=True):
        super().__init__()
        self.backbone = ConvBackbone(use_caam=use_caam)
        self.head = MFIHead(256, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for multi-class classification (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class UncertaintyWeighting(nn.Module):
    """Homoscedastic uncertainty-based adaptive task weighting.

    Learns log(sigma^2) for each task. The combined loss is:
        L = (1 / (2*sigma_1^2)) * L_1 + log(sigma_1)
          + (1 / (2*sigma_2^2)) * L_2 + log(sigma_2)

    Reference: Kendall, Gal & Cipolla, CVPR 2018.
    """

    def __init__(self, n_tasks=2):
        super().__init__()
        # Initialise log_vars to 0 -> sigma^2 = 1 -> equal weighting
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *losses):
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total
