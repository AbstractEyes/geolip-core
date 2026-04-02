"""
SVD observation — structural decomposition of feature maps and token sequences.

SVDObserver:      Spatial features (B, C, H, W) → projects via 1×1 conv, decomposes.
SVDTokenObserver: Token sequences (B, seq, dim) → transposes, decomposes directly.

Both extract: singular values, rotation structure (Vh), novelty (EMA deviation).
Both use geolip_core.linalg for decomposition — dispatches to cuSOLVER for training
(differentiable) or FL kernel for compiled inference (zero graph breaks).

Usage:
    from geolip_core.core.input.svd import SVDObserver, SVDTokenObserver

    spatial_obs = SVDObserver(in_channels=384, svd_rank=24)
    S, Vh, features, novelty = spatial_obs(conv_features)  # (B, C, H, W)

    token_obs = SVDTokenObserver(seq_len=5)
    S, Vh, features, novelty = token_obs(tokens)  # (B, 5, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import geolip_core.linalg as LA


class _SVDFeatureMixin:
    """Shared SVD feature extraction and EMA tracking."""

    def _init_ema(self, rank):
        self.register_buffer('ema_s', torch.ones(rank))
        self.register_buffer('ema_vh_flat', torch.eye(rank).reshape(-1))
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

    def _decompose(self, matrix):
        """SVD via geolip_core.linalg. (B, M, N) → S, Vh.

        Dispatches through LA.svd:
            method='gram_eigh' → Gram matrix + eigh (default, differentiable)
            Compiled inference automatically uses FL kernel (zero graph breaks).
        """
        with torch.amp.autocast('cuda', enabled=False):
            U, S, Vh = LA.svd(matrix.float(), method='gram_eigh')
            S = S.clamp(min=1e-6)
            S = torch.where(torch.isfinite(S), S, torch.ones_like(S))
            Vh = torch.where(torch.isfinite(Vh), Vh, torch.zeros_like(Vh))
        return S, Vh

    @torch.no_grad()
    def update_ema(self, S, Vh):
        """Update running averages. Call AFTER backward."""
        m = self.ema_momentum
        self.ema_s.mul_(m).add_(S.detach().mean(0), alpha=1-m)
        self.ema_vh_flat.mul_(m).add_(Vh.detach().mean(0).reshape(-1), alpha=1-m)


class SVDObserver(nn.Module, _SVDFeatureMixin):
    """Observe spatial backbone features via SVD decomposition.

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
        self._init_ema(svd_rank)

    def forward(self, x):
        """(B, C, H, W) → S, Vh, features, novelty."""
        B, C, H, W = x.shape
        h = self.to_svd(x)
        h_flat = h.permute(0, 2, 3, 1).reshape(B, H * W, self.svd_rank)
        S, Vh = self._decompose(h_flat)
        return S, Vh, self.extract_features(S, Vh), self.compute_novelty(S)

    @property
    def feature_dim(self):
        return 2 * self.svd_rank + 2


class SVDTokenObserver(nn.Module, _SVDFeatureMixin):
    """Observe token sequence structure via SVD decomposition.

    Transposes (B, seq, dim) → (B, dim, seq). The seq singular values
    tell you how the token directions distribute variance. Vh (seq×seq)
    tells you how token dimensions mix in feature space.

    No learned projection — the raw token structure is the observation.

    Args:
        seq_len: Number of tokens in input sequence
    """

    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self._init_ema(seq_len)

    def forward(self, x):
        """(B, seq, dim) → S, Vh, features, novelty."""
        x_t = x.transpose(1, 2).contiguous()
        S, Vh = self._decompose(x_t)
        return S, Vh, self.extract_features(S, Vh), self.compute_novelty(S)

    @property
    def feature_dim(self):
        return 2 * self.seq_len + 2