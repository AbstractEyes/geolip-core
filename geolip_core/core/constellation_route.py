"""
Constellation Route — ODE flow in tangent space.

Historical: superseded by ConstellationRelay for most use cases.
Retained for research and specialized applications.

Usage:
    from core.constellation_route import FlowAttention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowAttention(nn.Module):
    """3-step Euler flow in tangent space of S^(d-1).

    Routes embeddings through an ODE that integrates velocity fields
    conditioned on the triangulation profile. The correction is projected
    back to the tangent plane before normalization.

    Superseded by ConstellationRelay for production use (relay is faster,
    more stable, and achieves better fidelity at depth). Kept for research.

    Args:
        dim: embedding dimension
        n_anchors: constellation anchors for triangulation input
        flow_dim: internal flow dimension
        n_steps: Euler integration steps
        time_dim: sinusoidal time embedding dimension
        gate_init: initial gate logit
    """

    def __init__(self, dim, n_anchors, flow_dim=64, n_steps=3,
                 time_dim=32, gate_init=-3.0):
        super().__init__()
        self.dim = dim
        self.flow_dim = flow_dim
        self.n_anchors = n_anchors
        self.n_steps = n_steps
        self.time_dim = time_dim

        self.to_flow = nn.Sequential(
            nn.Linear(n_anchors + dim, flow_dim),
            nn.LayerNorm(flow_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, flow_dim),
            nn.GELU(),
        )
        self.stats_proj = nn.Linear(3, flow_dim, bias=False)
        self.velocity = nn.Sequential(
            nn.Linear(flow_dim, flow_dim * 2),
            nn.GELU(),
            nn.Linear(flow_dim * 2, flow_dim),
        )
        self.to_correction = nn.Linear(flow_dim, dim, bias=False)
        self.gate = nn.Parameter(torch.full((dim,), gate_init))
        self.register_buffer('stats_bias_cached', torch.zeros(flow_dim), persistent=False)

    def update_stats(self, push_diag, anchor_push):
        """Update cached stats bias from anchor push diagnostics."""
        with torch.no_grad():
            dev = self.stats_proj.weight.device
            if anchor_push.strategy == 'momentum' and anchor_push.accumulator is not None:
                mn = anchor_push.accumulator.norm(dim=-1)
            else:
                mn = torch.zeros(self.n_anchors, device=dev)
            dr = torch.tensor(push_diag.get('drift_mean', 0.0), device=dev).expand(self.n_anchors)
            ut = torch.tensor(push_diag.get('util_max', 0.0), device=dev).expand(self.n_anchors)
            self.stats_bias_cached = self.stats_proj(torch.stack([mn, ut, dr], -1)).mean(0)

    def forward(self, emb, constellation):
        """Apply ODE flow to embeddings using constellation anchors.

        Args:
            emb: (B, D) L2-normalized embeddings
            constellation: Constellation module (for anchors)

        Returns:
            (B, D) L2-normalized corrected embeddings
        """
        B, D, dev = *emb.shape, emb.device
        tri = emb @ F.normalize(constellation.anchors, dim=-1).T
        z = self.to_flow(torch.cat([tri, emb], -1))
        dt = 1.0 / self.n_steps
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=dev) / half)

        for s in range(self.n_steps):
            args = (s * dt) * freqs
            t_emb = torch.cat([args.sin(), args.cos()])
            z = z + dt * (self.velocity(z + self.time_mlp(t_emb)) + self.stats_bias_cached)

        # Project correction to tangent plane
        c = self.to_correction(z)
        c = c - (c * emb).sum(-1, keepdim=True) * emb
        return F.normalize(emb + torch.sigmoid(self.gate) * c, dim=-1)