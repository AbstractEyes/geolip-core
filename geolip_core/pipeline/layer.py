"""
ConstellationLayer — one depth of geometric observation.

Composes behaviors from core/:
    input.SVDObserver      → structural decomposition
    associate.Constellation → anchor frame triangulation
    curate.AnchorGate      → CM validity selection
    curate.Patchwork       → compartmentalized interpretation

Pipeline:
    features → OBSERVE (SVD)
             → ASSOCIATE (triangulate against anchors)
             → CURATE (CM gate + gated patchwork)
             → CONDITION (modulate backbone features)
             → geo_state residual (accumulate across depths)

Usage:
    from geolip_core.pipeline.layer import ConstellationLayer

    layer = ConstellationLayer(in_channels=256, spatial_size=8)
    x_mod, geo_state, obs = layer(features, geo_state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.core.input.svd import SVDObserver
from geolip_core.core.associate.constellation import init_anchors_repulsion
from geolip_core.core.curate.gate import AnchorGate
from geolip_core.core.util import make_activation


class ConstellationLayer(nn.Module):
    """One depth of geometric observation + curation.

    The constellation is the primary state. It owns the anchor positions
    on S^(d-1). Everything else reads from or conditions upon this frame.

    Args:
        in_channels:   Backbone feature channels at this depth
        spatial_size:  Spatial resolution (H=W) at this depth
        embed_dim:     Constellation embedding dimension
        n_anchors:     Number of anchors on S^(embed_dim - 1)
        n_comp:        Patchwork compartments
        d_comp:        Per-compartment hidden dim
        svd_rank:      SVD projection rank (≤32 for sub-ms)
        gate_strategy: 'round_robin', 'cm_gate', 'top_k', 'top_p'
        n_neighbors:   CM simplex neighbors
        activation:    Activation function name
    """

    def __init__(self, in_channels, spatial_size, embed_dim=256,
                 n_anchors=32, n_comp=8, d_comp=32,
                 svd_rank=24, gate_strategy='cm_gate', n_neighbors=3,
                 activation='gelu', top_k=None, top_p=0.9):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.svd_rank = svd_rank
        self.n_anchors = n_anchors
        pw_dim = n_comp * d_comp
        self.pw_dim = pw_dim

        # ── INPUT: SVD observation ──
        self.svd_observer = SVDObserver(in_channels, svd_rank)

        # SVD features → pw_dim projection
        self.svd_proj = nn.Sequential(
            nn.Linear(self.svd_observer.feature_dim, pw_dim),
            nn.LayerNorm(pw_dim), nn.GELU())

        # ── ASSOCIATE: Constellation embedding + anchors ──
        self.embed_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
        )
        self.anchors = nn.Parameter(init_anchors_repulsion(n_anchors, embed_dim))

        # ── CURATE: CM gate + patchwork compartments ──
        self.gate = AnchorGate(
            n_anchors, embed_dim, n_comp, n_neighbors,
            gate_strategy, top_k, top_p)

        self.comps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_anchors, d_comp * 2),
                make_activation(activation),
                nn.Linear(d_comp * 2, d_comp),
                nn.LayerNorm(d_comp))
            for _ in range(n_comp)])

        # ── CONDITION: modulate backbone features ──
        self.modulate = nn.Sequential(
            nn.Linear(pw_dim * 3, in_channels),
            nn.Sigmoid())

        # ── Geo residual stream ──
        self.geo_gate = nn.Parameter(torch.full((pw_dim,), -3.0))
        self.geo_proj = nn.Sequential(
            nn.Linear(pw_dim, pw_dim), nn.LayerNorm(pw_dim))

    def forward(self, x, geo_state):
        """Full geometric observation at one depth.

        Args:
            x:         (B, C, H, W) — backbone features
            geo_state: (B, pw_dim) — accumulated context from prior depths

        Returns:
            x_out:         (B, C, H, W) — modulated backbone features
            geo_state_out: (B, pw_dim) — updated geometric context
            obs:           dict — all intermediates for diagnostics
        """
        B, C, H, W = x.shape

        # ════ INPUT: SVD structural decomposition ════
        S, Vh, svd_raw, novelty = self.svd_observer(x)
        svd_context = self.svd_proj(svd_raw)

        # ════ ASSOCIATE: constellation triangulation ════
        emb = F.normalize(self.embed_proj(x), dim=-1)
        anchors_n = F.normalize(self.anchors, dim=-1)

        cos = emb @ anchors_n.detach().T
        tri = 1.0 - cos

        # ════ CURATE: CM gate + gated patchwork ════
        gate_values, gate_assign, gate_info = self.gate(emb, anchors_n.detach(), tri)
        tri_gated = tri * gate_values

        pw = torch.cat([comp(tri_gated) for comp in self.comps], dim=-1)

        # ════ CONDITION: modulate backbone features ════
        mod_input = torch.cat([pw, svd_context, geo_state], dim=-1)
        channel_weights = self.modulate(mod_input)
        x_out = x * channel_weights.unsqueeze(-1).unsqueeze(-1)

        # ════ GEO RESIDUAL ════
        g = torch.sigmoid(self.geo_gate)
        geo_state_out = geo_state + g * self.geo_proj(pw)

        obs = {
            'embedding': emb, 'cos': cos, 'tri': tri,
            'patchwork': pw, 'singular_values': S, 'Vh': Vh,
            'svd_context': svd_context, 's_deviation': novelty,
            'gate_info': gate_info, 'gate_values': gate_values,
        }

        return x_out, geo_state_out, obs

    @torch.no_grad()
    def update_ema(self, S, Vh):
        """Update SVD running averages. Call AFTER backward."""
        self.svd_observer.update_ema(S, Vh)
