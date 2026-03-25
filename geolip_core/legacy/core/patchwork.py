"""
Patchwork — round-robin compartmentalized Curation of triangulation.
MagnitudeFlow — relay-stack per-compartment magnitude prediction (Mutation).
AnchorPush — non-gradient anchor repositioning strategies.

In the six-stage paradigm:
  Patchwork     = Curation (interprets association distances into features)
  MagnitudeFlow = Mutation (transforms magnitude informed by triangulation)
  AnchorPush    = external to the paradigm (non-gradient, between epochs)

Usage:
    from geolip_core.core.patchwork import Patchwork, MagnitudeFlow, AnchorPush
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import make_activation
from .constellation_relay import RelayLayer


# ── PATCHWORK ──

class Patchwork(nn.Module):
    """Round-robin compartments reading diverse anchor subsets.

    Assigns anchors to compartments via modular arithmetic (anchor_i → comp_{i%n_comp}).
    Each compartment is an independent MLP reading only its assigned anchor distances.
    Output is the concatenation of all compartment outputs.

    Args:
        n_anchors: total anchors in constellation
        n_comp: number of compartments (default 8)
        d_comp: output dimension per compartment (default 64)
        activation: activation function name
    """

    def __init__(self, n_anchors, n_comp=8, d_comp=64, activation='squared_relu'):
        super().__init__()
        self.n_comp, self.d_comp = n_comp, d_comp
        self.output_dim = n_comp * d_comp
        self.register_buffer('asgn', torch.arange(n_anchors) % n_comp)
        apc = n_anchors // n_comp
        self.comps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(apc, d_comp * 2), make_activation(activation),
                nn.Linear(d_comp * 2, d_comp), nn.LayerNorm(d_comp),
            )
            for _ in range(n_comp)
        ])

    def forward(self, tri):
        """tri: (B, n_anchors) → (B, n_comp * d_comp)."""
        return torch.cat([
            self.comps[k](tri[:, self.asgn == k])
            for k in range(self.n_comp)
        ], dim=-1)


# ── MAGNITUDE FLOW ──

class MagnitudeFlow(nn.Module):
    """Relay-stack per-compartment magnitude prediction. No attention.

    Reads embedding + triangulation + raw magnitude through relay layers,
    then predicts per-compartment magnitude scalars that modulate the
    triangulation before patchwork reads it.

    Args:
        dim: embedding dimension
        n_anchors: constellation anchors
        hidden_dim: relay patchwork hidden dim
        n_heads: unused (API compat)
        n_layers: number of relay layers
        mag_min: minimum magnitude output
        mag_max: maximum magnitude output
        n_comp: number of patchwork compartments
    """

    def __init__(self, dim, n_anchors, hidden_dim=64, n_heads=4,
                 n_layers=2, mag_min=0.1, mag_max=5.0, n_comp=8):
        super().__init__()
        self.dim, self.n_anchors = dim, n_anchors
        self.mag_min, self.mag_max = mag_min, mag_max
        self.n_comp, self.n_layers = n_comp, n_layers
        patch_dim = 16
        relay_dim = n_comp * patch_dim
        self.patch_dim, self.relay_dim = patch_dim, relay_dim

        self.emb_proj = nn.Linear(dim, relay_dim // 2)
        self.tri_proj = nn.Linear(n_anchors, relay_dim // 4)
        self.ctx_proj = nn.Linear(relay_dim // 2 + relay_dim // 4 + 1, relay_dim)
        self.relays = nn.ModuleList([
            RelayLayer(relay_dim, patch_dim, 16, 3, hidden_dim, -3.0)
            for _ in range(n_layers)
        ])
        self.mag_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_dim, patch_dim // 2), nn.GELU(),
                nn.Linear(patch_dim // 2, 1),
            )
            for _ in range(n_comp)
        ])
        self.register_buffer('stats_bias_cached', torch.zeros(n_comp), persistent=False)

    def update_stats(self, push_diag, anchor_push):
        """Update cached stats bias from anchor push diagnostics."""
        with torch.no_grad():
            device = self.stats_bias_cached.device
            if anchor_push.strategy == 'momentum' and anchor_push.accumulator is not None:
                mn = anchor_push.accumulator.norm(dim=-1)
                apc = self.n_anchors // self.n_comp
                self.stats_bias_cached = torch.stack([
                    mn[k * apc: (k + 1) * apc if k < self.n_comp - 1 else self.n_anchors].mean()
                    for k in range(self.n_comp)
                ])
            else:
                self.stats_bias_cached.zero_()

    def forward(self, emb, triangulation, raw_magnitude):
        """Predict per-anchor magnitude from context.

        Args:
            emb: (B, D) L2-normalized embedding
            triangulation: (B, A) cosine similarities to anchors
            raw_magnitude: (B, 1) pre-normalization feature magnitude

        Returns:
            mag: (B, A) per-anchor magnitude
            mag_comp: (B, n_comp) per-compartment magnitude
        """
        B, A = emb.shape[0], self.n_anchors
        x = self.ctx_proj(torch.cat([
            self.emb_proj(emb), self.tri_proj(triangulation), raw_magnitude
        ], -1))
        for relay in self.relays:
            x = relay(x)
        patches = x.reshape(B, self.n_comp, self.patch_dim)
        mc = torch.cat([self.mag_heads[k](patches[:, k]) for k in range(self.n_comp)], -1)
        mc = self.mag_min + (self.mag_max - self.mag_min) * torch.sigmoid(mc + self.stats_bias_cached)
        apc = A // self.n_comp
        mag = torch.cat([
            mc[:, k:k + 1].expand(-1, apc if k < self.n_comp - 1 else A - k * apc)
            for k in range(self.n_comp)
        ], -1)
        return mag, mc

    def get_relay_diagnostics(self):
        """Return drift and gate stats for each relay layer."""
        return [
            {
                'layer': i,
                'drift_mean': r.drift().mean().item(),
                'gate_mean': r.gates.sigmoid().mean().item(),
            }
            for i, r in enumerate(self.relays)
        ]


# ── ANCHOR PUSH ──

def _project_tangent(vec, point):
    """Project vec onto tangent plane of S^(d-1) at point."""
    return vec - (vec * point).sum(dim=-1, keepdim=True) * point


def _compute_centroids_and_assign(anchors_n, emb_n, label_buffer, device):
    """Compute class centroids and assign anchors to classes via greedy matching."""
    n_a = anchors_n.shape[0]
    classes = label_buffer.unique()
    n_cls = classes.shape[0]
    centroids = torch.cat([
        F.normalize(emb_n[label_buffer == c].mean(0, keepdim=True), dim=-1)
        for c in classes if (label_buffer == c).sum() > 0
    ], dim=0)
    if centroids.shape[0] == 0:
        return None, None, None, None

    cos = anchors_n @ centroids.T
    apc = n_a // n_cls
    assigned = torch.full((n_a,), -1, dtype=torch.long, device=device)
    cc = torch.zeros(n_cls, dtype=torch.long, device=device)
    for idx in cos.flatten().sort(descending=True).indices:
        a, c = (idx // n_cls).item(), (idx % n_cls).item()
        if assigned[a] >= 0 or cc[c] >= apc + 1:
            continue
        assigned[a] = c
        cc[c] += 1
        if (assigned >= 0).all():
            break
    u = (assigned < 0).nonzero(as_tuple=True)[0]
    if len(u) > 0:
        assigned[u] = (anchors_n[u] @ centroids.T).argmax(1)

    nearest = (emb_n @ anchors_n.T).argmax(1)
    util = torch.bincount(nearest, minlength=n_a).float()
    return centroids, assigned, util / util.sum().clamp(min=1), classes


def _perturb_target(target, apc, rank):
    """Add slight noise to target for anchors sharing a class centroid."""
    if apc > 1 and rank > 0:
        noise = torch.randn_like(target) * 0.05
        return F.normalize(target + noise - (noise * target).sum() * target, dim=-1)
    return target


class AnchorPush:
    """Configurable non-gradient anchor repositioning.

    Strategies:
        raw: direct SLERP toward class centroid
        momentum: tangent-space momentum with dead anchor revival
        gru: GRU-style gating (historical)

    Args:
        strategy: 'raw', 'momentum', or 'gru'
        n_anchors: number of anchors
        dim: embedding dimension
        **kw: strategy-specific kwargs
    """

    def __init__(self, strategy, n_anchors, dim, **kw):
        self.strategy = strategy
        self.n_anchors = n_anchors
        self.dim = dim
        self.push_count = 0

        if strategy == 'raw':
            self.lr = kw.get('lr', 0.1)
        elif strategy == 'momentum':
            self.decay = kw.get('decay', 0.9)
            self.alpha = kw.get('alpha', 0.1)
            self.beta = kw.get('beta', 0.05)
            self.util_floor = kw.get('util_floor', 0.001)
            self.accumulator = None
        elif strategy == 'gru':
            self.ema_decay = kw.get('ema_decay', 0.9)
            self.z_scale = kw.get('z_scale', 3.0)
            self.r_scale = kw.get('r_scale', 5.0)
            self.prev_pos = self.util_ema = self.drift_ema = None

    @torch.no_grad()
    def push(self, core, emb_buf, lbl_buf):
        """Push anchors toward class centroids.

        Args:
            core: module with .constellation.anchors (nn.Parameter)
            emb_buf: (N, D) accumulated embeddings
            lbl_buf: (N,) corresponding labels

        Returns:
            dict: diagnostics (drift, utilization, etc.)
        """
        anchors = core.constellation.anchors.data
        n_a = anchors.shape[0]
        device = anchors.device
        emb_n = F.normalize(emb_buf, dim=-1)
        anchors_n = F.normalize(anchors, dim=-1)

        centroids, assigned, util, classes = _compute_centroids_and_assign(
            anchors_n, emb_n, lbl_buf, device)
        if centroids is None:
            return {'moved': 0}

        if hasattr(core, 'anchor_classes'):
            for a in range(n_a):
                core.anchor_classes[a] = classes[assigned[a]]
        if hasattr(core, 'class_centroids'):
            for i, c in enumerate(classes):
                core.class_centroids[c] = centroids[i]

        apc = n_a // centroids.shape[0]
        targets = torch.stack([
            _perturb_target(
                centroids[assigned[a].item()], apc,
                (assigned[:a] == assigned[a]).sum().item()
            )
            for a in range(n_a)
        ])

        if self.strategy == 'raw':
            for a in range(n_a):
                anchors[a] = F.normalize(
                    anchors_n[a] + self.lr * (targets[a] - anchors_n[a]), dim=-1)
            d = torch.acos(
                (anchors_n * F.normalize(anchors, dim=-1)).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
            diag = {'drift_mean': d.mean().item(), 'drift_max': d.max().item()}

        elif self.strategy == 'momentum':
            if self.accumulator is None:
                self.accumulator = torch.zeros(n_a, self.dim, device=device)
            res = _project_tangent(targets - anchors_n, anchors_n)
            self.accumulator = self.decay * _project_tangent(self.accumulator, anchors_n) + res
            corr = self.alpha * res + self.beta * self.accumulator
            dead = util < self.util_floor
            if dead.any():
                corr[dead] = res[dead] * 0.5
            new = F.normalize(anchors_n + corr, dim=-1)
            d = torch.acos((anchors_n * new).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
            anchors.copy_(new)
            diag = {
                'drift_mean': d.mean().item(), 'drift_max': d.max().item(),
                'momentum_mean': self.accumulator.norm(dim=-1).mean().item(),
                'dead_count': dead.sum().item(),
            }
        else:
            diag = {}

        diag.update({
            'moved': n_a,
            'n_active': (util > 0).sum().item(),
            'util_min': util.min().item(),
            'util_max': util.max().item(),
        })
        self.push_count += 1
        return diag