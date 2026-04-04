"""
PositionGeometricContext — 5-stream fusion → FiLM context.

Five streams:
    anchor:     cos_to_anchors + assignment + triangulation — WHERE on the manifold
    structural: patchwork + embedding — WHAT the local geometry looks like
    history:    geo_residual from previous layers — WHAT prior layers observed
    quality:    CM gate values per anchor — HOW TRUSTWORTHY is this observation
    flow:       FlowEnsemble predictions — WHAT other mathematical lenses see
"""

import torch
import torch.nn as nn
from geolip_core.pipeline.observer import TorchComponent


class PositionGeometricContext(TorchComponent):
    """Curation stage: 5-stream fusion → FiLM context.

    The flow stream starts at zero (zero-init) and learns to contribute.
    Without flows attached, the 5th stream is zeros — equivalent to the
    original 4-stream architecture.
    """
    def __init__(self, name, n_anchors, pw_dim, manifold_dim, context_dim):
        super().__init__(name)
        self.context_dim = context_dim
        self.pw_dim = pw_dim
        self.manifold_dim = manifold_dim

        # WHERE on the manifold
        self.anchor_mlp = nn.Sequential(
            nn.Linear(n_anchors * 3, context_dim), nn.GELU(), nn.LayerNorm(context_dim))
        # WHAT the local geometry looks like
        self.struct_mlp = nn.Sequential(
            nn.Linear(pw_dim + manifold_dim, context_dim), nn.GELU(), nn.LayerNorm(context_dim))
        # WHAT prior layers observed
        self.history_mlp = nn.Sequential(
            nn.Linear(pw_dim, context_dim), nn.GELU(), nn.LayerNorm(context_dim))
        # HOW TRUSTWORTHY — full per-anchor gate profile
        self.quality_mlp = nn.Sequential(
            nn.Linear(n_anchors, context_dim), nn.GELU(), nn.LayerNorm(context_dim))
        # FLOW OPINIONS — anchor-space flow ensemble [N, A] (same shape as gate_values)
        # Small init: negligible contribution at start, nonzero gradient path
        self.flow_mlp = nn.Sequential(
            nn.Linear(n_anchors, context_dim), nn.GELU(), nn.LayerNorm(context_dim))
        nn.init.normal_(self.flow_mlp[0].weight, std=0.01)
        nn.init.zeros_(self.flow_mlp[0].bias)

        # Fuse 5 streams
        self.fuse = nn.Sequential(
            nn.Linear(context_dim * 5, context_dim), nn.GELU(), nn.LayerNorm(context_dim))

    def forward(self, obs_dict, gate_values=None, geo_residual=None, flow_output=None):
        """
        Args:
            obs_dict: from decomposed association + gated curation
            gate_values: (N, A) CM gate values per anchor, or None
            geo_residual: (N, pw_dim) accumulated context, or None for first layer
            flow_output: (N, manifold_dim) flow ensemble prediction, or None
        Returns:
            (N, context_dim) geometric context for FiLM
        """
        anchor_feats = torch.cat([
            obs_dict['cos_to_anchors'],
            obs_dict['assignment'],
            obs_dict['triangulation'],
        ], dim=-1)
        struct_feats = torch.cat([
            obs_dict['patchwork'],
            obs_dict['embedding'],
        ], dim=-1)

        a = self.anchor_mlp(anchor_feats)
        s = self.struct_mlp(struct_feats)
        h = self.history_mlp(geo_residual) if geo_residual is not None else torch.zeros_like(a)
        q = self.quality_mlp(gate_values) if gate_values is not None else torch.zeros_like(a)
        f = self.flow_mlp(flow_output) if flow_output is not None else torch.zeros_like(a)

        return self.fuse(torch.cat([a, s, h, q, f], dim=-1))
