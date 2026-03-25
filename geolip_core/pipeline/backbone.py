"""
GeometricBackbone — multi-depth geometric observation stack.

Wraps any feature-producing backbone with ConstellationLayers at
specified depths. Threads geo_state residual across layers.

This is a PIPELINE — it composes ConstellationLayers from pipeline/layer.py
into a multi-depth system. The behaviors themselves live in core/.

Usage:
    from geolip_core.pipeline.backbone import GeometricBackbone

    geo = GeometricBackbone(stages=[(64, 32), (128, 16), (256, 8), (384, 4)])
    modulated, geo_state, obs = geo(features_per_depth)
"""

import torch
import torch.nn as nn

from .layer import ConstellationLayer


class GeometricBackbone(nn.Module):
    """Multi-depth geometric observation stack.

    Each depth gets independent: SVD observer, constellation anchors,
    CM gate, and patchwork compartments. The geo_state residual stream
    carries accumulated geometric context across depths.

    Args:
        stages:        List of (in_channels, spatial_size) per depth
        embed_dim:     Constellation embedding dimension
        n_anchors:     Anchors per depth
        n_comp:        Patchwork compartments
        d_comp:        Per-compartment dim
        svd_rank:      SVD projection rank
        gate_strategy: CM gate strategy
        n_neighbors:   CM simplex neighbors
        activation:    Activation function name
    """

    def __init__(self, stages, embed_dim=256, n_anchors=32,
                 n_comp=8, d_comp=32, svd_rank=24,
                 gate_strategy='cm_gate', n_neighbors=3,
                 activation='gelu'):
        super().__init__()
        self.n_depths = len(stages)
        pw_dim = n_comp * d_comp
        self.pw_dim = pw_dim

        self.layers = nn.ModuleList([
            ConstellationLayer(
                in_channels=ch, spatial_size=sp,
                embed_dim=embed_dim, n_anchors=n_anchors,
                n_comp=n_comp, d_comp=d_comp,
                svd_rank=svd_rank, gate_strategy=gate_strategy,
                n_neighbors=n_neighbors, activation=activation)
            for ch, sp in stages])

    def forward(self, features_per_depth, geo_state=None):
        """Run geometric observation at each depth.

        Args:
            features_per_depth: list of (B, C, H, W) tensors, one per depth
            geo_state: (B, pw_dim) or None (initialized to zeros)

        Returns:
            modulated_features: list of (B, C, H, W)
            geo_state: (B, pw_dim) — final geometric context
            all_obs: list of dicts — per-depth diagnostics
        """
        B = features_per_depth[0].shape[0]
        device = features_per_depth[0].device

        if geo_state is None:
            geo_state = torch.zeros(B, self.pw_dim, device=device)

        modulated = []
        all_obs = []

        for layer, feat in zip(self.layers, features_per_depth):
            feat_mod, geo_state, obs = layer(feat, geo_state)
            modulated.append(feat_mod)
            all_obs.append(obs)

        return modulated, geo_state, all_obs

    @torch.no_grad()
    def update_emas(self, all_obs):
        """Update all depth EMAs after backward."""
        for layer, obs in zip(self.layers, all_obs):
            layer.update_ema(obs['singular_values'], obs['Vh'])
