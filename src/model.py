"""MT-OPMNet model architecture (paper Section IV).

Four-block 1-D CNN backbone with MaxPool, Channel-Aware Attention Module
(CAAM, reduction=8), and two task-specific heads for OSNR regression and
modulation format classification.
"""

import torch
import torch.nn as nn


class CAAM(nn.Module):
    """Channel-Aware Attention Module (paper Section IV-B).

    SE-style squeeze-and-excitation with reduction ratio r=8.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class ConvBlock(nn.Module):
    """1-D Conv block: Conv1D → BatchNorm → ReLU → MaxPool(2).

    No dropout in backbone (dropout is heads-only per paper).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MTOPMNet(nn.Module):
    """Multi-Task Optical Performance Monitoring Network (paper Section IV-A).

    Architecture:
        4 × ConvBlock (32→64→128→256, kernel=5, MaxPool)
        → CAAM (reduction=8)
        → Flatten
        → OSNR head (FC 128→64→1, dropout 0.3)
        → MFI head  (FC 128→64→K, dropout 0.3)

    Args:
        n_bins: Number of input histogram bins (100).
        n_classes: Number of modulation format classes (5).
        use_caam: Whether to apply CAAM.
        caam_reduction: CAAM bottleneck reduction ratio (8).
        dropout: Dropout rate for task heads (0.3).
    """

    def __init__(self, n_bins: int = 100, n_classes: int = 5,
                 use_caam: bool = True, caam_reduction: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        self.use_caam = use_caam

        # Shared backbone: four Conv1D blocks (paper Table II)
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, kernel=5),
            ConvBlock(32, 64, kernel=5),
            ConvBlock(64, 128, kernel=5),
            ConvBlock(128, 256, kernel=5),
        )

        # CAAM after final conv block
        if use_caam:
            self.caam = CAAM(256, reduction=caam_reduction)

        # Compute flattened dimension: n_bins after 4× MaxPool(2)
        flat_dim = 256 * (n_bins // 16)

        # OSNR regression head (paper Section IV-C)
        self.osnr_head = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # MFI classification head (paper Section IV-C)
        self.mfi_head = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation (paper Algorithm 1, line 1)."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (batch, 1, n_bins).

        Returns:
            osnr: (batch, 1), mfi_logits: (batch, n_classes).
        """
        features = self.backbone(x)

        if self.use_caam:
            features = self.caam(features)

        flat = features.view(features.size(0), -1)

        osnr = self.osnr_head(flat)
        mfi_logits = self.mfi_head(flat)

        return osnr, mfi_logits
