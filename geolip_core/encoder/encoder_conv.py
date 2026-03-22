"""
ConvEncoder — 8-layer conv backbone for image classification.

Proven on CIFAR-10/100. Simple, no attention, no geometric layers.
Just feature extraction into a flat vector on S^(d-1).

Usage:
    from encoder.encoder_conv import ConvEncoder
"""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """8-layer conv → D-dim projection.

    Architecture: 4 blocks of (conv-BN-GELU, conv-BN-GELU, MaxPool)
    Channels: 64 → 128 → 256 → 384
    Output: (B, output_dim) after linear + LayerNorm

    Note: L2 normalization is NOT applied here — the caller decides
    when to normalize (preserving raw magnitude for MagnitudeFlow).

    Args:
        output_dim: embedding dimension (default 256)
    """

    def __init__(self, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 384, 3, padding=1), nn.BatchNorm2d(384), nn.GELU(),
            nn.Conv2d(384, 384, 3, padding=1), nn.BatchNorm2d(384), nn.GELU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(384, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        """x: (B, 3, H, W) → (B, output_dim) unnormalized features."""
        return self.proj(self.features(x))