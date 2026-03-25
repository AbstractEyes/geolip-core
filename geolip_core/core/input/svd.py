"""
SVD observation — structural decomposition of feature maps.

Observes a backbone's internal state by projecting features to a
low-rank subspace and decomposing via SVD. Extracts energy distribution
(singular values), rotation structure (Vh), and novelty (EMA deviation).

Uses utils/kernel.py for the actual decomposition. This module defines
the OBSERVATION BEHAVIOR — what to project, what features to extract,
how to track structural change over time.

Usage:
    from geolip_core.core.input.svd import SVDObserver

    obs = SVDObserver(in_channels=384, svd_rank=24)
    S, Vh, features, novelty = obs(conv_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.utils.kernel import gram_eigh_svd


class SVDObserver(nn.Module):
    """Observe backbone features via SVD decomposition.

    Projects in_channels to svd_rank via 1×1 conv, decomposes the
    spatial structure, extracts compact features, tracks novelty via EMA.

    Args:
        in_channels: Backbone feature channels
        svd_rank: SVD projection rank (≤32 for sub-ms)
    """

    def __init__(self, in_channels, svd_rank=24):
        super().__init__()
        self.svd_rank = svd_rank
        self.to_svd = nn.Conv2d(in_channels, svd_rank, 1, bias=False)

        # EMA tracking for novelty detection
        self.register_buffer('ema_s', torch.ones(svd_rank))
        self.register_buffer('ema_vh_flat', torch.eye(svd_rank).reshape(-1))
        self.ema_momentum = 0.99

    def extract_features(self, S, Vh):
        """Compact SVD summary. (B, k), (B, k, k) → (B, 2k+2)."""
        B, k = S.shape
        S_safe = S.clamp(min=1e-6)
        s_norm = S_safe / (S_safe.sum(dim=-1, keepdim=True) + 1e-8)
        vh_diag = Vh.diagonal(dim1=-2, dim2=-1)
        vh_offdiag = (Vh.pow(2).sum((-2, -1)) - vh_diag.pow(2).sum(-1)).unsqueeze(-1).clamp(min=0)
        s_ent = -(s_norm * torch.log(s_norm.clamp(min=1e-8))).sum(-1, keepdim=True)
        out = torch.cat([s_norm, vh_diag, vh_offdiag, s_ent], dim=-1)
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    def compute_novelty(self, S):
        """Deviation from running average. (B, k) → (B, k)."""
        return S - self.ema_s.clone().unsqueeze(0)

    def forward(self, x):
        """Observe backbone features.

        Args:
            x: (B, C, H, W) backbone feature map

        Returns:
            S:        (B, k) singular values
            Vh:       (B, k, k) rotation matrix
            features: (B, 2k+2) compact SVD summary
            novelty:  (B, k) deviation from EMA
        """
        B, C, H, W = x.shape
        h = self.to_svd(x)  # (B, k, H, W)
        h_flat = h.permute(0, 2, 3, 1).reshape(B, H * W, self.svd_rank)

        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                _, S, Vh = gram_eigh_svd(h_flat.float())
                S = S.clamp(min=1e-6)
                S = torch.where(torch.isfinite(S), S, torch.ones_like(S))
                Vh = torch.where(torch.isfinite(Vh), Vh, torch.zeros_like(Vh))

        features = self.extract_features(S, Vh)
        novelty = self.compute_novelty(S)

        return S, Vh, features, novelty

    @torch.no_grad()
    def update_ema(self, S, Vh):
        """Update running averages. Call AFTER backward."""
        m = self.ema_momentum
        self.ema_s.mul_(m).add_(S.detach().mean(0), alpha=1-m)
        self.ema_vh_flat.mul_(m).add_(Vh.detach().mean(0).reshape(-1), alpha=1-m)

    @property
    def feature_dim(self):
        """Output feature dimension: 2*svd_rank + 2."""
        return 2 * self.svd_rank + 2
