"""
Constellation Relay — per-token geometric Mutation.

RelayLayer:         Single vectorized relay. Patches → S^(d-1) → 3-phase SLERP → gated residual.
ConstellationRelay: Per-token wrapper. O(S) complexity. 99.4% cos fidelity at depth 16.

In the six-stage paradigm, these are Mutations — they transform position
on the manifold informed by triangulation, without changing what the
embedding represents.

Usage:
    from geolip_core.core.constellation_relay import ConstellationRelay, RelayLayer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RELAY LAYER ──

class RelayLayer(nn.Module):
    """Single constellation relay. Vectorized, gated, no attention.

    Splits input into patches, normalizes each to S^(patch_dim-1),
    triangulates against learned anchors at 3 SLERP phases (0, 1/3, 2/3),
    then applies per-patch patchwork MLP with gated residual.

    The home/anchors pair enables stroboscopic multi-phase triangulation:
    anchors SLERP between their home position and current learned position.

    Args:
        input_dim: total input dimension (must be divisible by patch_dim)
        patch_dim: dimension per patch (default 16, the natural S^15 dimension)
        n_anchors: anchors per patch
        n_phases: SLERP phases (default 3)
        pw_hidden: patchwork MLP hidden dim
        gate_init: initial gate logit (negative = conservative start)
    """

    def __init__(self, input_dim, patch_dim=16, n_anchors=16,
                 n_phases=3, pw_hidden=32, gate_init=-3.0):
        super().__init__()
        assert input_dim % patch_dim == 0
        self.input_dim, self.patch_dim = input_dim, patch_dim
        self.n_patches = input_dim // patch_dim
        self.n_anchors, self.n_phases = n_anchors, n_phases
        P, A, d = self.n_patches, n_anchors, patch_dim

        # Home position (frozen reference) + current anchors (learned)
        home = torch.empty(P, A, d)
        nn.init.xavier_normal_(home.view(P * A, d))
        home = F.normalize(home.view(P, A, d), dim=-1)
        self.register_buffer('home', home)
        self.anchors = nn.Parameter(home.clone())

        # Per-patch patchwork MLP (vectorized via einsum)
        tri_dim = n_phases * A
        self.pw_w1 = nn.Parameter(torch.empty(P, tri_dim, pw_hidden))
        self.pw_b1 = nn.Parameter(torch.zeros(1, P, pw_hidden))
        self.pw_w2 = nn.Parameter(torch.empty(P, pw_hidden, d))
        self.pw_b2 = nn.Parameter(torch.zeros(1, P, d))
        for p in range(P):
            nn.init.xavier_normal_(self.pw_w1.data[p])
            nn.init.xavier_normal_(self.pw_w2.data[p])
        self.pw_norm = nn.LayerNorm(d)
        self.gates = nn.Parameter(torch.full((P,), gate_init))
        self.norm = nn.LayerNorm(input_dim)

    def drift(self):
        """Angular drift from home position (radians). Shape: (P, A)."""
        h = F.normalize(self.home, dim=-1)
        c = F.normalize(self.anchors, dim=-1)
        return torch.acos((h * c).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7))

    def at_phase(self, t):
        """SLERP between home and current at phase t ∈ [0, 1]. Shape: (P, A, d)."""
        h = F.normalize(self.home, dim=-1)
        c = F.normalize(self.anchors, dim=-1)
        omega = self.drift().unsqueeze(-1)
        so = omega.sin().clamp(min=1e-7)
        return torch.sin((1 - t) * omega) / so * h + torch.sin(t * omega) / so * c

    def forward(self, x):
        """x: (B, D) → (B, D) with gated geometric residual."""
        B, D = x.shape
        P, A, d = self.n_patches, self.n_anchors, self.patch_dim

        patches = self.norm(x).reshape(B, P, d)
        patches_n = F.normalize(patches, dim=-1)

        # Multi-phase triangulation
        tris = []
        for t in [0.0, 1 / 3, 2 / 3]:
            at = F.normalize(self.at_phase(t), dim=-1)
            tris.append(1.0 - torch.einsum('bpd,pad->bpa', patches_n, at))
        tri = torch.cat(tris, dim=-1)

        # Per-patch patchwork MLP
        h = F.gelu(torch.einsum('bpt,pth->bph', tri, self.pw_w1) + self.pw_b1)
        pw = self.pw_norm(torch.einsum('bph,phd->bpd', h, self.pw_w2) + self.pw_b2)

        # Gated residual
        gate = self.gates.sigmoid().unsqueeze(0).unsqueeze(-1)
        return x + (gate * pw + (1 - gate) * patches).reshape(B, D)


# ── CONSTELLATION RELAY (sequence-aware wrapper) ──

class ConstellationRelay(nn.Module):
    """Per-token geometric processing. O(S) complexity.

    Drop-in replacement for attention layers. Handles both
    (B, D) flat and (B, S, D) sequential inputs.

    Args:
        dim: token dimension
        n_anchors: constellation anchors
        n_comp: patchwork compartments
        d_comp: per-compartment hidden dim
        gate_init: initial gate logit
        anchor_init: anchor initialization strategy
        activation: activation function name
    """

    def __init__(self, dim, n_anchors=16, n_comp=8, d_comp=64,
                 gate_init=-3.0, anchor_init='repulsion', activation='squared_relu'):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # Lazy import to avoid circular dependency
        from .constellation import Constellation
        from .patchwork import Patchwork

        self.constellation = Constellation(n_anchors, dim, anchor_init=anchor_init)
        self.patchwork = Patchwork(n_anchors, n_comp, d_comp, activation)
        self.proj = nn.Linear(self.patchwork.output_dim, dim)
        self.gate = nn.Parameter(torch.full((dim,), gate_init))

    def forward(self, x, return_tri=False):
        """x: (B, D) or (B, S, D) → same shape, with geometric residual.

        Args:
            x: input tensor
            return_tri: if True, also return triangulation profile for routing

        Returns:
            out: same shape as x
            tri (optional): (B, S, n_anchors) triangulation distances if return_tri=True
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        B, S, D = x.shape
        residual = x

        h_flat = F.normalize(self.norm(x).reshape(B * S, D), dim=-1)
        tri, _ = self.constellation.triangulate(h_flat)
        update = self.proj(self.patchwork(tri)).reshape(B, S, D)
        out = residual + torch.sigmoid(self.gate) * update

        if squeeze:
            out = out.squeeze(1)
            if return_tri:
                return out, tri  # (B, D), (B, n_anchors)
            return out

        if return_tri:
            return out, tri.reshape(B, S, -1)  # (B, S, D), (B, S, n_anchors)
        return out