"""MT-OPMNet model architecture with Channel-Aware Attention Module (CAAM)."""

import torch
import torch.nn as nn


class CAAM(nn.Module):
    """Channel-Aware Attention Module.

    Performs adaptive feature recalibration by learning channel-wise
    importance weights via a squeeze-and-excitation style mechanism.
    """

    def __init__(self, channels: int, reduction: int = 4):
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
    """1-D convolution block: Conv1D -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MTOPMNet(nn.Module):
    """Multi-Task Optical Performance Monitoring Network.

    Shared 1-D CNN backbone with optional CAAM attention, feeding into
    two task-specific heads for OSNR regression and modulation format
    classification.

    Args:
        n_bins: Number of input amplitude histogram bins.
        n_classes: Number of modulation format classes.
        use_caam: Whether to apply Channel-Aware Attention Module.
    """

    def __init__(self, n_bins: int = 100, n_classes: int = 5,
                 use_caam: bool = True):
        super().__init__()
        self.use_caam = use_caam

        # Shared backbone: three Conv1D blocks with increasing channels
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, kernel=7),
            ConvBlock(32, 64, kernel=5),
            ConvBlock(64, 128, kernel=3),
        )

        # Optional channel-aware attention
        if use_caam:
            self.caam = CAAM(128)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # OSNR regression head
        self.osnr_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # Modulation format classification head
        self.mfi_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming initialisation for Conv/Linear, constant for BN."""
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
            x: Input tensor of shape (batch, 1, n_bins).

        Returns:
            osnr: Predicted OSNR values, shape (batch, 1).
            mfi_logits: Modulation format logits, shape (batch, n_classes).
        """
        features = self.backbone(x)

        if self.use_caam:
            features = self.caam(features)

        pooled = self.pool(features).squeeze(-1)

        osnr = self.osnr_head(pooled)
        mfi_logits = self.mfi_head(pooled)

        return osnr, mfi_logits
