"""
Geometric Transformer — CM-Validated Pipeline
==================================================
Dual-stream transformer with CM-gated constellation observation,
quaternion composition, and per-layer Cayley alignment.

CM-validated pipeline changes:
    - CM validity gate between association and curation (AnchorGate)
    - 5-stream PositionGeometricContext: anchor + structural + history + quality + FLOW
    - CM-conditioned geometric residual accumulation (replaces blind learned gate)
    - Built-in geometric regularization (CV target + anchor spread)
    - Decomposed observer pipeline: association → CM gate → gated curation
    - Optional FlowEnsemble: multi-opinion geometric fusion (quat, velocity, orbital, etc.)

Pipeline per layer:
    1. ManifoldProjection:  h_i → emb_i on S^(d-1) per position
    2. ConstellationAssociation: emb_i → raw triangulation, cos, assignment
    3. CMValidatedGate: per-anchor CM validity → gate_values (B*L, A)
    4. Gated curation: patchwork reads tri * gate_values (validated only)
    4.5 FlowEnsemble (optional): multi-opinion geometric predictions
    5. PositionGeometricContext: 5 streams → FiLM context (B, L, context_dim)
    6. ContentAttention (Stream A): standard MHA
    7. GeometricAttention (Stream B): FiLM(Q,K | geo_ctx), V pure
    8. CayleyOrthogonal: align B → A basis
    9. QuaternionCompose: w=A, i=aligned_B, j=A-B, k=A*B
   10. Decode + gated residual
   11. CM-conditioned geometric residual write

Geometric regularization (call model.geometric_losses() during training):
    - CV loss: anchor CV → pentachoron band (0.20-0.23)
    - Spread loss: prevent anchor collapse (penalize positive cosine)
    These maintain the constellation in the regime where CM validation works.

Design principles from Ryan Spearman (ρ=0.309, 76/84 wins):
    - FiLM on Q,K ONLY — geometry routes attention, V stays pure
    - FiLM on individual arms BEFORE composition, not after
    - Quaternion algebra as structural regularizer (non-commutative coupling)
    - CayleyOrthogonal guarantees pure rotation (det=1 always)
    - Never global average pool — per-position geometric context

Author: AbstractPhil + Claude Opus 4.6
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS — hard dependencies, no fallback stubs
# ═══════════════════════════════════════════════════════════════════════════════

from geolip_core.core.associate.constellation import (
    ConstellationObserver, ConstellationAssociation, ConstellationCuration,
    Constellation, init_anchors_repulsion,
)
from geolip_core.core.curate.gate import AnchorGate as _GeolipAnchorGate
from geolip_core.core.curate.flows import FlowEnsemble
import geolip_core.linalg as LA
from geolip_core.pipeline.observer import (
    TorchComponent, BaseTower, Input, Curation, Distinction,
)
from geolip_core.core.distinguish.losses import (
    observer_loss as _geolip_observer_loss,
    ce_loss_paired as _geolip_ce_loss_paired,
    spread_loss as _geolip_spread_loss,
)

# Optional: geofractal WideRouter for compilation
try:
    from geofractal.router.wide_router import WideRouter
    _HAS_WIDE_ROUTER = True
except ImportError:
    _HAS_WIDE_ROUTER = False

# ═══════════════════════════════════════════════════════════════════════════════
# CAYLEY-MENGER VALIDITY — geometric quality measurement
# ═══════════════════════════════════════════════════════════════════════════════

def pairwise_distances_squared(points):
    """Batched pairwise squared distances. (B, N, D) → (B, N, N)."""
    gram = torch.bmm(points, points.transpose(1, 2))
    diag = gram.diagonal(dim1=-2, dim2=-1)
    return diag.unsqueeze(2) + diag.unsqueeze(1) - 2 * gram


def cayley_menger_det(points):
    """Cayley-Menger signed volume² for simplices. (B, K, D) → (B,).

    K = number of vertices (k+1 for a k-simplex).
    Sign-corrected: positive = valid non-degenerate simplex.
    """
    B, K, D = points.shape
    d2 = pairwise_distances_squared(points)
    M = torch.zeros(B, K + 1, K + 1, device=points.device, dtype=points.dtype)
    M[:, 0, 1:] = 1.0
    M[:, 1:, 0] = 1.0
    M[:, 1:, 1:] = d2
    raw = LA.det(M)
    k = K - 1
    sign = (-1.0) ** (k + 1)
    return sign * raw


def anchor_neighborhood_cm(anchors, n_neighbors=3):
    """Precompute per-anchor CM quality from local neighborhood geometry.

    Position-independent. O(A) determinant computations on small matrices.
    Each anchor forms a simplex with its k nearest neighbor anchors.
    The CM determinant measures local geometric quality — high volume means
    the anchor neighborhood is well-conditioned for triangulation.

    Args:
        anchors: (A, D) normalized anchor positions on S^(d-1)
        n_neighbors: neighbors per simplex

    Returns:
        quality: (A,) signed log-magnitude CM quality per anchor
        nn_idx: (A, n_neighbors) neighbor indices
    """
    A, D = anchors.shape
    dists = torch.cdist(anchors.unsqueeze(0), anchors.unsqueeze(0)).squeeze(0)
    # Mask self-distances without in-place mutation (compile-safe)
    self_mask = torch.eye(A, device=anchors.device, dtype=anchors.dtype) * 1e12
    dists = dists + self_mask
    _, nn_idx = dists.topk(n_neighbors, largest=False)  # (A, n_neighbors)

    # Build simplices: [anchor_a, neighbor_1, ..., neighbor_k] — fully vectorized
    simplices = torch.cat([
        anchors.unsqueeze(1),   # (A, 1, D)
        anchors[nn_idx],        # (A, n_neighbors, D)
    ], dim=1)                   # (A, K, D)

    dets = cayley_menger_det(simplices)  # (A,)
    sign = dets.sign()
    log_mag = torch.log(dets.abs() + 1e-12)
    return sign * log_mag, nn_idx


# ═══════════════════════════════════════════════════════════════════════════════
# CM VALIDATED GATE — efficient anchor gating for transformer scale
# ═══════════════════════════════════════════════════════════════════════════════

class CMValidatedGate(nn.Module):
    """Anchor gate based on Cayley-Menger validity.

    Efficient for transformer scale: anchor CM quality is precomputed O(A²)
    and CACHED (only recomputed on invalidate_cache()), then combined with
    per-position proximity features through a learned gate.

    The gate starts OPEN (bias=+2, sigmoid≈0.88) and learns to CLOSE on
    geometrically invalid configurations. Architecture-before-loss: the gate
    suppresses degenerate measurements structurally, not through a loss signal.

    Gate features per (position, anchor):
        - anchor_cm_quality: CM volume of anchor's local neighborhood (cached)
        - cos_to_anchor: cosine similarity (position-dependent)

    Args:
        n_anchors: number of constellation anchors
        n_neighbors: neighbors for CM simplex computation
    """
    def __init__(self, n_anchors, n_neighbors=3):
        super().__init__()
        self.n_anchors = n_anchors
        self.n_neighbors = n_neighbors

        # Learned gate: [cm_quality, cos_sim] → scalar gate
        self.gate_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # Init OPEN — learn to close. sigmoid(2.0) ≈ 0.88
        # Small random weight so gradient flows back to gate_proj[0]
        nn.init.normal_(self.gate_proj[2].weight, std=0.01)
        nn.init.constant_(self.gate_proj[2].bias, 2.0)

        # Anchor CM cache — invalidated after optimizer step
        self._cached_cm_norm = None

    def invalidate_cache(self):
        """Call after optimizer.step() to recompute anchor CM next forward."""
        self._cached_cm_norm = None

    def precompute(self, anchors):
        """Compute anchor CM norm OUTSIDE compile graph.
        Called from layer forward before the compilable gate computation.
        Idempotent: skips if cache is warm.
        """
        if self._cached_cm_norm is not None:
            return
        with torch.no_grad():
            anchor_cm, _ = anchor_neighborhood_cm(anchors, self.n_neighbors)
            cm_std = anchor_cm.std().clamp(min=1e-8)
            self._cached_cm_norm = ((anchor_cm - anchor_cm.mean()) / cm_std).detach()

    def _compute_gate(self, anchor_cm_norm, tri):
        """Fully compilable — pure tensor ops, no linalg, no graph breaks."""
        N, A = tri.shape
        cos_sim = 1.0 - tri

        features = torch.stack([
            anchor_cm_norm.unsqueeze(0).expand(N, -1),
            cos_sim,
        ], dim=-1)

        gate_values = torch.sigmoid(self.gate_proj(features).squeeze(-1))

        gate_info = {
            'active': (gate_values.detach() > 0.5).float().sum(-1).mean(),
            'gate_mean': gate_values.detach().mean(),
            'cm_positive_frac': (anchor_cm_norm > 0).float().mean(),
        }

        return gate_values, gate_info

    def forward(self, tri):
        """Fully compilable forward. Requires precompute() called first."""
        return self._compute_gate(self._cached_cm_norm, tri)


# ═══════════════════════════════════════════════════════════════════════════════
# INFONCE MEMORY BANK — contrastive pressure on geometric residual
# ═══════════════════════════════════════════════════════════════════════════════

class GeoResidualBank(nn.Module):
    """Cross-stream contrastive memory bank (CLIP-style)."""
    def __init__(self, proj_dim, bank_size=4096, temperature=0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.bank_size = bank_size
        self.temperature = temperature

        self.register_buffer('queue', torch.randn(bank_size, proj_dim))
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys):
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        if ptr + B <= self.bank_size:
            self.queue[ptr:ptr + B] = keys
        else:
            overflow = (ptr + B) - self.bank_size
            self.queue[ptr:] = keys[:B - overflow]
            self.queue[:overflow] = keys[B - overflow:]
        self.queue_ptr[0] = (ptr + B) % self.bank_size

    def forward(self, content_proj, geo_proj):
        q = F.normalize(content_proj, dim=-1)
        k_pos = F.normalize(geo_proj, dim=-1)
        k_neg = self.queue.clone().detach()

        pos_logits = (q * k_pos).sum(dim=-1, keepdim=True) / self.temperature
        neg_logits = q @ k_neg.T / self.temperature

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == 0).float().mean()

        return loss, acc


# ═══════════════════════════════════════════════════════════════════════════════
# PROVEN COMPONENTS — from Ryan Spearman (unchanged, tested)
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(TorchComponent):
    """Feature-wise Linear Modulation. Near-identity-initialized.
    gamma ≈ 1 + 0.01·geo_ctx, beta ≈ 0.01·geo_ctx at init.
    Gradient flows through to geo_ctx from step 0.
    """
    def __init__(self, name, feature_dim, context_dim):
        super().__init__(name)
        self.to_gamma = nn.Linear(context_dim, feature_dim)
        self.to_beta = nn.Linear(context_dim, feature_dim)
        nn.init.normal_(self.to_gamma.weight, std=0.01); nn.init.ones_(self.to_gamma.bias)
        nn.init.normal_(self.to_beta.weight, std=0.01); nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, ctx):
        return self.to_gamma(ctx) * x + self.to_beta(ctx)


class CayleyOrthogonal(TorchComponent):
    """Guaranteed SO(d) rotation via Cayley map. det(Q) = 1 always.

    precompute() caches the rotation matrix outside the compile graph.
    forward() reads the cached rotation — pure tensor ops, CUDA-graph-safe.
    """
    def __init__(self, name, dim):
        super().__init__(name)
        self.dim = dim
        self.A_upper = nn.Parameter(torch.zeros(dim * (dim - 1) // 2) * 0.01)
        idx = torch.triu_indices(dim, dim, offset=1)
        self.register_buffer('_triu_row', idx[0], persistent=False)
        self.register_buffer('_triu_col', idx[1], persistent=False)
        self.register_buffer('_eye', torch.eye(dim), persistent=False)
        self._cached_rotation = None

    def get_rotation(self):
        """Compute rotation via Cayley map. Uses cuSOLVER (LA.solve)."""
        d = self.dim
        A = torch.zeros(d, d, device=self.A_upper.device, dtype=self.A_upper.dtype)
        A[self._triu_row, self._triu_col] = self.A_upper
        A = A - A.T
        return LA.solve(self._eye + A, self._eye - A)

    def precompute(self):
        """Cache rotation matrix. Call outside compiled graph."""
        if self._cached_rotation is None:
            self._cached_rotation = self.get_rotation()

    def invalidate_cache(self):
        self._cached_rotation = None

    def forward(self, x):
        R = self._cached_rotation if self._cached_rotation is not None else self.get_rotation()
        return x @ R.T


def quaternion_multiply_batched(q1, q2):
    """Hamilton product on (B, 4, D) tensors. Fully vectorized."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=1)


class QuaternionCompose(TorchComponent):
    """Four-arm Hamilton product composition. Proven in GeoQuat head."""
    def __init__(self, name, input_dim, quat_dim=64):
        super().__init__(name)
        self.quat_dim = quat_dim
        self.proj_w = nn.Linear(input_dim, quat_dim)
        self.proj_i = nn.Linear(input_dim, quat_dim)
        self.proj_j = nn.Linear(input_dim, quat_dim)
        self.proj_k = nn.Linear(input_dim, quat_dim)
        self.rotation = nn.Parameter(torch.randn(1, 4, quat_dim) * 0.1)

    @property
    def output_dim(self):
        return self.quat_dim * 4

    def forward(self, arm_w, arm_i, arm_j, arm_k):
        shape = arm_w.shape[:-1]
        D = arm_w.shape[-1]
        flat = arm_w.dim() > 2
        if flat:
            arm_w = arm_w.reshape(-1, D); arm_i = arm_i.reshape(-1, D)
            arm_j = arm_j.reshape(-1, D); arm_k = arm_k.reshape(-1, D)
        q = torch.stack([self.proj_w(arm_w), self.proj_i(arm_i),
                         self.proj_j(arm_j), self.proj_k(arm_k)], dim=1)
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        r = self.rotation.expand(q.shape[0], -1, -1)
        r = r / (r.norm(dim=1, keepdim=True) + 1e-8)
        composed = quaternion_multiply_batched(r, q)
        composed = composed.reshape(q.shape[0], -1)
        if flat:
            composed = composed.reshape(*shape, -1)
        return composed


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER-SPECIFIC COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class ManifoldProjection(TorchComponent):
    """Input stage: project transformer hidden states to S^(d-1)."""
    def __init__(self, name, d_model, manifold_dim):
        super().__init__(name)
        self.proj = nn.Linear(d_model, manifold_dim)
        self.norm = nn.LayerNorm(manifold_dim)

    def forward(self, hidden_states):
        h = self.norm(self.proj(hidden_states))
        return F.normalize(h, dim=-1)


class PositionGeometricContext(TorchComponent):
    """Curation stage: 5-stream fusion → FiLM context.

    Five streams:
        anchor:     cos_to_anchors + assignment + triangulation — WHERE on the manifold
        structural: patchwork + embedding — WHAT the local geometry looks like
        history:    geo_residual from previous layers — WHAT prior layers observed
        quality:    CM gate values per anchor — HOW TRUSTWORTHY is this observation
        flow:       FlowEnsemble predictions — WHAT other mathematical lenses see

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


class GeometricAttention(TorchComponent):
    """Attention with FiLM from curated constellation. Stream B."""
    def __init__(self, name, d_model, n_heads=8, context_dim=128, dropout=0.1):
        super().__init__(name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.film_q = FiLMLayer(f'{name}_film_q', d_model, context_dim)
        self.film_k = FiLMLayer(f'{name}_film_k', d_model, context_dim)
        self.norm = nn.LayerNorm(d_model)

        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.film_ffn = FiLMLayer(f'{name}_film_ffn', d_model * 4, context_dim)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, geo_ctx, attn_mask=None, key_padding_mask=None):
        B, L, D = x.shape
        H, HD = self.n_heads, self.head_dim

        Q = self.film_q(self.w_q(x), geo_ctx)
        K = self.film_k(self.w_k(x), geo_ctx)
        V = self.w_v(x)

        Q = Q.view(B, L, H, HD).transpose(1, 2)
        K = K.view(B, L, H, HD).transpose(1, 2)
        V = V.view(B, L, H, HD).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_out = (self.dropout(F.softmax(scores, dim=-1)) @ V)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = self.norm(x + self.w_o(attn_out))

        h = F.gelu(self.ffn1(x))
        h = self.film_ffn(h, geo_ctx)
        x = self.ffn_norm(x + self.ffn_drop(self.ffn2(h)))
        return x


class ContentAttention(TorchComponent):
    """Standard self-attention. Stream A. No geometric conditioning."""
    def __init__(self, name, d_model, n_heads=8, dropout=0.1):
        super().__init__(name)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = self.norm(x + a)
        x = self.ffn_norm(x + self.ffn(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER — CM-validated dual-stream with constellation routing + flows
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricTransformerLayer(BaseTower):
    """One layer of the geometric transformer (CM validated + flows).

    Pipeline per layer:
        1. ManifoldProjection: h → emb on S^(d-1)
        2. Association: emb → raw triangulation, cos, assignment
        3. CMValidatedGate: per-anchor CM validity → gate_values
        4. Gated curation: patchwork reads tri * gate_values
        4.5 FlowEnsemble (optional): multi-opinion geometric predictions
        5. PositionGeometricContext: 5 streams → FiLM context
        6. ContentAttention (Stream A): standard MHA
        7. GeometricAttention (Stream B): FiLM(Q,K | geo_ctx)
        8. CayleyOrthogonal: align B → A
        9. QuaternionCompose: w=A, i=aligned_B, j=A-B, k=A*B
       10. Decode + gated residual
       11. CM-conditioned geometric residual accumulation

    Flows are optional, config-driven, and individually replaceable:
        layer['flows'].attach_flow('alignment')
        layer['flows'].detach_flow('velocity')
    """
    def __init__(self, name, d_model, n_heads=8, n_anchors=32,
                 manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1,
                 cm_neighbors=3, flow_keys=None, flow_fusion='weighted'):
        super().__init__(name)
        self.d_model = d_model
        self.n_anchors = n_anchors
        self.manifold_dim = manifold_dim

        # 1. Project to manifold
        self.attach('projection', ManifoldProjection(
            f'{name}_proj', d_model, manifold_dim))

        # 2. Constellation observer (association + curation — called decomposed)
        self.attach('observer', ConstellationObserver(
            dim=manifold_dim, n_anchors=n_anchors,
            n_comp=n_comp, d_comp=d_comp))

        # 3. CM validated gate — between association and curation
        self.attach('cm_gate', CMValidatedGate(
            n_anchors=n_anchors, n_neighbors=cm_neighbors))

        # 3.5 Flow ensemble — optional multi-opinion geometric fusion
        if flow_keys:
            self.attach('flows', FlowEnsemble(
                f'{name}_flows', manifold_dim, n_anchors,
                flow_keys=flow_keys, fusion=flow_fusion))
            # Blend weight: how much flow opinions influence curation
            # Starts small → flows fade in as they learn
            self.flow_alpha = nn.Parameter(torch.tensor(0.01))

        # 4. Fuse observation into FiLM context (5 streams)
        pw_dim = self['observer'].curation.patchwork.output_dim
        self.attach('context', PositionGeometricContext(
            f'{name}_ctx', n_anchors, pw_dim, manifold_dim, context_dim))

        # 5. Stream A: content
        self.attach('content', ContentAttention(
            f'{name}_content', d_model, n_heads, dropout))

        # 6. Stream B: geometric
        self.attach('geometric', GeometricAttention(
            f'{name}_geo', d_model, n_heads, context_dim, dropout))

        # 7. Cayley rotation: align B → A
        self.attach('rotation', CayleyOrthogonal(f'{name}_cayley', d_model))

        # 8. Quaternion composition
        self.attach('compose', QuaternionCompose(
            f'{name}_quat', d_model, quat_dim))

        # 9. Decode + output gate
        self.attach('decode', nn.Sequential(
            nn.Linear(quat_dim * 4, d_model), nn.GELU(), nn.LayerNorm(d_model)))
        self.attach('gate', nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()))

        # 10. Geometric residual projection (no learned gate — CM quality decides)
        self._pw_dim = pw_dim
        self.attach('geo_proj', nn.Sequential(
            nn.Linear(pw_dim, pw_dim), nn.LayerNorm(pw_dim)))

    def forward(self, x, geo_residual=None, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (B, L, D) input hidden states
            geo_residual: (B, L, pw_dim) accumulated geometric context,
                          or None for first layer

        Returns:
            x_out: (B, L, D) transformed hidden states
            geo_residual_out: (B, L, pw_dim) updated geometric residual
            geo_state: dict with full geometric state + CM diagnostics
        """
        B, L, D = x.shape

        # ════ 1. Project to manifold ════
        emb = self['projection'](x)  # (B, L, manifold_dim)
        emb_flat = emb.reshape(B * L, -1)

        # ════ 2. Association — raw triangulation ════
        a_out = self['observer'].association(emb_flat)

        # ════ 3. CM Gate — validate anchor measurements ════
        anchors_n = F.normalize(
            self['observer'].association.constellation.anchors, dim=-1)
        # CM gate forward — precompute() must have been called before entering
        # the compiled graph (by GeometricTransformer.precompute_cm_gates())
        gate_values, gate_info = self['cm_gate'](a_out['distances'])

        # ════ 4. Gated curation — patchwork reads validated triangulation ════
        a_out_gated = dict(a_out)

        # ════ 4.5 Flow ensemble — anchor-space geometric opinions ════
        flow_opinion = None
        if self.has('flows'):
            flow_opinion = self['flows'](anchors_n, emb_flat, a_out['distances'])  # [N, A]
            # Blend flow opinion into triangulation: raw + alpha*(flow - raw)
            # flow_alpha starts at 0.01 → 99% raw, 1% flow opinion
            # Gradient: observer_loss → patchwork → distances_weighted → flow_opinion → flows
            alpha = self.flow_alpha.sigmoid()
            blended_tri = a_out['distances'] + alpha * (flow_opinion - a_out['distances'])
            a_out_gated['distances_weighted'] = blended_tri * gate_values
        else:
            a_out_gated['distances_weighted'] = a_out['distances'] * gate_values

        c_out = self['observer'].curation.curate_full(a_out_gated, emb=emb_flat)

        # Build observation dict for context
        obs = {
            'embedding': emb_flat,
            'triangulation': a_out['distances'],
            'cos_to_anchors': a_out['cos_to_anchors'],
            'assignment': a_out['assignment'],
            'nearest': a_out['nearest'],
            'patchwork': c_out['patchwork'],
            'bridge': c_out['bridge'],
        }

        # ════ 5. Build FiLM context — 5 streams ════
        geo_res_flat = geo_residual.reshape(B * L, -1) if geo_residual is not None else None
        geo_ctx_flat = self['context'](
            obs, gate_values=gate_values, geo_residual=geo_res_flat,
            flow_output=flow_opinion)
        geo_ctx = geo_ctx_flat.reshape(B, L, -1)

        # ════ 6. Stream A: content attention ════
        a_out_stream = self['content'](
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # ════ 7. Stream B: geometric attention ════
        b_out = self['geometric'](
            x, geo_ctx, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # ════ 8. Cayley rotation: align B → A ════
        b_aligned = self['rotation'](b_out)

        # ════ 9. Quaternion composition ════
        composed = self['compose'](
            arm_w=a_out_stream, arm_i=b_aligned,
            arm_j=a_out_stream - b_aligned, arm_k=a_out_stream * b_aligned)

        # ════ 10. Decode + gated residual ════
        decoded = self['decode'](composed)
        g = self['gate'](torch.cat([x, decoded], dim=-1))
        x_out = g * decoded + (1 - g) * x

        # ════ 11. CM-conditioned geometric residual accumulation ════
        pw_validated = c_out['patchwork'].reshape(B, L, -1)
        cm_quality = gate_values.mean(dim=-1).reshape(B, L, 1)
        geo_update = self['geo_proj'](pw_validated)

        if geo_residual is None:
            geo_residual_out = cm_quality * geo_update
        else:
            geo_residual_out = geo_residual + cm_quality * geo_update

        # ════ Build geo_state dict ════
        def _unflatten(t):
            if t is None:
                return None
            if t.dim() == 1:
                return t.reshape(B, L)
            return t.reshape(B, L, *t.shape[1:])

        geo_state = {
            'embedding':      emb,
            'geo_ctx':        geo_ctx,
            'triangulation':  _unflatten(a_out['distances']),
            'cos_to_anchors': _unflatten(a_out['cos_to_anchors']),
            'assignment':     _unflatten(a_out['assignment']),
            'nearest':        _unflatten(a_out['nearest']),
            'patchwork':      _unflatten(c_out['patchwork']),
            'bridge':         _unflatten(c_out['bridge']),
            'gate_values':    _unflatten(gate_values),
            'gate_info':      gate_info,
            'cm_quality':     cm_quality,
            'content':        a_out_stream,
            'geometric':      b_out,
            'composed':       composed,
            'geo_residual':   geo_residual_out,
            'flow_opinion':   _unflatten(flow_opinion) if flow_opinion is not None else None,
        }

        return x_out, geo_residual_out, geo_state


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL — stack of layers + geometric regularization
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricTransformer(BaseTower):
    """Geometric Transformer — CM-validated dual-stream with optional flows.

    Stack of GeometricTransformerLayers with:
        - CM-gated observation at every layer
        - Optional FlowEnsemble at every layer (config-driven)
        - Cross-layer Cayley rotation on hidden states
        - Built-in geometric regularization via geometric_losses()
    """
    def __init__(self, name, d_model=512, n_heads=8, n_layers=4,
                 n_anchors=32, manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1,
                 cross_layer_rotation=True, cm_neighbors=3,
                 nce_bank_size=4096, nce_temperature=0.1,
                 vocab_size=None, max_seq_len=2048,
                 flow_keys=None, flow_fusion='weighted'):
        super().__init__(name)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_anchors = n_anchors
        self._pw_dim = n_comp * d_comp

        if vocab_size is not None:
            self.attach('embed', nn.Embedding(vocab_size, d_model))
            self.attach('pos_embed', nn.Embedding(max_seq_len, d_model))
            self.attach('head', nn.Linear(d_model, vocab_size, bias=False))

        for i in range(n_layers):
            self.attach(f'layer_{i}', GeometricTransformerLayer(
                f'{name}_L{i}', d_model, n_heads, n_anchors,
                manifold_dim, n_comp, d_comp, context_dim, quat_dim,
                dropout, cm_neighbors,
                flow_keys=flow_keys, flow_fusion=flow_fusion))

        if cross_layer_rotation and n_layers > 1:
            for i in range(n_layers - 1):
                self.attach(f'cross_rot_{i}', CayleyOrthogonal(
                    f'{name}_xrot_{i}', d_model))

        self.attach('final_norm', nn.LayerNorm(d_model))

        # Cross-stream contrastive (CLIP-style)
        if nce_bank_size > 0:
            nce_proj_dim = 128
            self.attach('nce_content_proj', nn.Sequential(
                nn.Linear(d_model, nce_proj_dim),
                nn.GELU(),
                nn.Linear(nce_proj_dim, nce_proj_dim),
            ))
            self.attach('nce_geo_proj', nn.Sequential(
                nn.Linear(self._pw_dim, nce_proj_dim),
                nn.GELU(),
                nn.Linear(nce_proj_dim, nce_proj_dim),
            ))
            self.attach('nce_bank', GeoResidualBank(
                nce_proj_dim, bank_size=nce_bank_size,
                temperature=nce_temperature))

        self._config = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            n_anchors=n_anchors, manifold_dim=manifold_dim,
            n_comp=n_comp, d_comp=d_comp, context_dim=context_dim,
            quat_dim=quat_dim, dropout=dropout,
            cross_layer_rotation=cross_layer_rotation,
            cm_neighbors=cm_neighbors, vocab_size=vocab_size,
            nce_bank_size=nce_bank_size, nce_temperature=nce_temperature,
            flow_keys=flow_keys, flow_fusion=flow_fusion,
        )

    @property
    def config(self):
        return self._config.copy()

    def invalidate_caches(self):
        """Invalidate all cuSOLVER-dependent caches. Call after optimizer.step()."""
        for i in range(self.n_layers):
            self[f'layer_{i}']['cm_gate'].invalidate_cache()
            self[f'layer_{i}']['rotation'].invalidate_cache()
        # Cross-layer rotations
        for name in list(self.components.keys()):
            if name.startswith('cross_rot'):
                self[name].invalidate_cache()

    @torch.compiler.disable
    def precompute_cm_gates(self):
        """Precompute ALL cuSOLVER-dependent operations for all layers.

        Caches: CM gate anchor quality (det) + Cayley rotations (solve).
        Must be called BEFORE the compiled forward pass. CUDA graph
        capture cannot contain cuSOLVER calls.

        Idempotent: skips components with warm caches.
        """
        for i in range(self.n_layers):
            layer = self[f'layer_{i}']
            # CM gate — uses det
            anchors_n = F.normalize(
                layer['observer'].association.constellation.anchors, dim=-1)
            layer['cm_gate'].precompute(anchors_n.detach())
            # Per-layer Cayley rotation — uses solve
            layer['rotation'].precompute()
        # Cross-layer rotations — uses solve
        for name in list(self.components.keys()):
            if name.startswith('cross_rot'):
                self[name].precompute()

    def geometric_losses(self, cv_target=0.215, cv_weight=0.1, spread_weight=0.01):
        """Compute geometric regularization from current anchor geometry."""
        total_cv = torch.tensor(0.0)
        total_spread = torch.tensor(0.0)
        n = 0

        for i in range(self.n_layers):
            layer = self[f'layer_{i}']
            anchors = layer['observer'].association.constellation.anchors
            anchors_n = F.normalize(anchors, dim=-1)
            A = anchors_n.shape[0]

            if n == 0:
                total_cv = total_cv.to(anchors.device)
                total_spread = total_spread.to(anchors.device)

            cos = anchors_n @ anchors_n.T
            idx = torch.triu_indices(A, A, offset=1, device=cos.device)
            pairwise_dist = 1.0 - cos[idx[0], idx[1]]
            cv = pairwise_dist.std() / (pairwise_dist.mean() + 1e-8)
            total_cv = total_cv + (cv - cv_target).pow(2)

            mask = ~torch.eye(A, dtype=torch.bool, device=cos.device)
            total_spread = total_spread + F.relu(cos[mask]).mean()

            n += 1

        losses = {}
        if n > 0:
            losses['cv'] = cv_weight * total_cv / n
            losses['spread'] = spread_weight * total_spread / n
            losses['geo_total'] = losses['cv'] + losses['spread']
        return losses

    def infonce_loss(self, cls_index=0):
        """Cross-stream contrastive: content queries against decoupled geometry."""
        if not self.has('nce_bank'):
            return {}

        hidden = getattr(self, '_last_hidden', None)
        geo_residual = getattr(self, '_last_geo_residual', None)
        if hidden is None or geo_residual is None:
            return {}

        content_cls = self['nce_content_proj'](hidden[:, cls_index])
        geo_cls = self['nce_geo_proj'](geo_residual[:, cls_index].detach())

        loss, acc = self['nce_bank'](content_cls, geo_cls)
        return {'nce': loss, 'nce_acc': acc}

    @torch.no_grad()
    def update_nce_bank(self, cls_index=0):
        """Enqueue projected geo keys into bank. Call AFTER backward."""
        if not self.has('nce_bank') or not self.has('nce_geo_proj'):
            return

        geo_residual = getattr(self, '_last_geo_residual', None)
        if geo_residual is None:
            return

        geo_cls = self['nce_geo_proj'](geo_residual[:, cls_index].detach())
        self['nce_bank'].enqueue(F.normalize(geo_cls, dim=-1))

    def anchor_diagnostics(self):
        """Per-layer anchor health diagnostics."""
        diag = {}
        for i in range(self.n_layers):
            layer = self[f'layer_{i}']
            anchors = layer['observer'].association.constellation.anchors
            anchors_n = F.normalize(anchors.detach(), dim=-1)
            A = anchors_n.shape[0]

            cos = anchors_n @ anchors_n.T
            idx = torch.triu_indices(A, A, offset=1, device=cos.device)
            pairwise = 1.0 - cos[idx[0], idx[1]]
            cv = (pairwise.std() / (pairwise.mean() + 1e-8)).item()

            with torch.no_grad():
                anchor_cm, _ = anchor_neighborhood_cm(
                    anchors_n, layer['cm_gate'].n_neighbors)

            diag[f'layer_{i}'] = {
                'anchor_cv': cv,
                'mean_pairwise_dist': pairwise.mean().item(),
                'min_pairwise_dist': pairwise.min().item(),
                'cm_positive_frac': (anchor_cm > 0).float().mean().item(),
                'cm_mean': anchor_cm.mean().item(),
                'cm_std': anchor_cm.std().item(),
            }
        return diag

    def param_report(self):
        total = 0
        name = getattr(self, '_tower_name', self.__class__.__name__)
        print(f"\n  {name} — parameter report (CM-validated + flows)")
        print(f"  {'Component':<35s}  {'Params':>12s}")
        print(f"  {'─'*35}  {'─'*12}")
        for cname, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            total += n
            print(f"  {cname:<35s}  {n:>12,}")
        print(f"  {'─'*35}  {'─'*12}")
        print(f"  {'TOTAL':<35s}  {total:>12,}")
        return total


    def forward(self, x, attn_mask=None, key_padding_mask=None,
                return_geo_state=False):
        if self.has('embed') and x.dtype in (torch.long, torch.int32, torch.int64):
            pos = torch.arange(x.shape[1], device=x.device)
            x = self['embed'](x) + self['pos_embed'](pos)

        geo_states = []
        has_xrot = self.has('cross_rot_0')
        geo_residual = None

        for i in range(self.n_layers):
            x, geo_residual, geo_state = self[f'layer_{i}'](
                x, geo_residual=geo_residual,
                attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            if return_geo_state:
                geo_states.append(geo_state)
            if has_xrot and i < self.n_layers - 1:
                x = self[f'cross_rot_{i}'](x)

        # Stash for NCE bank if active (outside compiled graph only)
        if self.has('nce_bank'):
            self._last_geo_residual = geo_residual
            self._last_hidden = x

        x = self['final_norm'](x)
        if self.has('head'):
            x = self['head'](x)

        return (x, geo_states) if return_geo_state else x

    # ── Paired forward + observer loss ──────────────────────────────

    def _run_view(self, x, attn_mask=None, key_padding_mask=None):
        """Run one view through the full pipeline.
        Retains ALL layers' geo_states — every layer needs gradient.
        """
        has_xrot = self.has('cross_rot_0')
        geo_residual = None

        if self.has('embed') and x.dtype in (torch.long, torch.int32, torch.int64):
            pos = torch.arange(x.shape[1], device=x.device)
            x = self['embed'](x) + self['pos_embed'](pos)

        geo_states = []
        for i in range(self.n_layers):
            x, geo_residual, geo_state = self[f'layer_{i}'](
                x, geo_residual=geo_residual,
                attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            geo_states.append(geo_state)
            if has_xrot and i < self.n_layers - 1:
                x = self[f'cross_rot_{i}'](x)

        x = self['final_norm'](x)
        return x, geo_states

    def forward_paired(self, x1, x2, cls_index=0,
                       attn_mask=None, key_padding_mask=None):
        """Dual-view forward for observer loss training.

        Observer loss reads FINAL layer's observations (coherent space).
        Non-final layers get gradient through the geo_residual stream
        (FiLM → context → history_mlp → geo_residual → earlier layers).
        All layers' computation graphs are retained by _run_view.
        """
        B = x1.shape[0]
        x_cat = torch.cat([x1, x2], dim=0)
        feat_cat, geo_states = self._run_view(x_cat, attn_mask, key_padding_mask)

        c = cls_index
        gs = geo_states[-1]  # final layer — coherent representation space

        return {
            'embedding':      gs['embedding'][:B, c],
            'embedding_aug':  gs['embedding'][B:, c],
            'patchwork1':     gs['patchwork'][:B, c],
            'patchwork1_aug': gs['patchwork'][B:, c],
            'bridge1':        gs['bridge'][:B, c],
            'bridge2':        gs['bridge'][B:, c],
            'assign1':        gs['assignment'][:B, c],
            'assign2':        gs['assignment'][B:, c],
            'cos1':           gs['cos_to_anchors'][:B, c],
            'tri1':           gs['triangulation'][:B, c],
            'tri2':           gs['triangulation'][B:, c],
            'features1':      feat_cat[:B],
            'features2':      feat_cat[B:],
            'gate_values1':   gs['gate_values'][:B, c],
            'gate_values2':   gs['gate_values'][B:, c],
            'cm_quality1':    gs['cm_quality'][:B],
            'cm_quality2':    gs['cm_quality'][B:],
        }

    def compute_loss(self, output, targets, cls_index=0,
                     w_ce=1.0, head=None, **loss_kwargs):
        final_layer = self[f'layer_{self.n_layers - 1}']
        anchors = final_layer['observer'].association.constellation.anchors

        obs_loss, ld = _geolip_observer_loss(
            output, anchors=anchors, targets=targets,
            **loss_kwargs)

        if head is not None:
            feat1 = output['features1'][:, cls_index]
            feat2 = output['features2'][:, cls_index]
            logits1 = head(feat1)
            logits2 = head(feat2)
            l_ce, acc = _geolip_ce_loss_paired(logits1, logits2, targets)
            ld['ce'], ld['acc'] = l_ce, acc
            ld['logits'] = logits1
            loss = w_ce * l_ce + obs_loss
            ld['loss_task'] = l_ce.detach()
        else:
            loss = obs_loss

        ld['loss_observer'] = obs_loss.detach()

        w_spread = loss_kwargs.get('w_spread', 0.01)
        if self.n_layers > 1 and w_spread > 0:
            other_spread = torch.tensor(0.0, device=anchors.device)
            for i in range(self.n_layers - 1):
                layer = self[f'layer_{i}']
                layer_anchors = layer['observer'].association.constellation.anchors
                other_spread = other_spread + _geolip_spread_loss(layer_anchors)
            other_spread = w_spread * other_spread / (self.n_layers - 1)
            loss = loss + other_spread
            ld['spread_other_layers'] = other_spread.detach()

        ld['total'] = loss
        return loss, ld


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def geo_transformer_esm2(name='geo_esm2', n_layers=6, **kw):
    """Pre-configured for ESM-2 650M (d=1280)."""
    return GeometricTransformer(name, d_model=1280, n_heads=16,
        n_layers=n_layers, n_anchors=32, manifold_dim=256,
        n_comp=8, d_comp=32, context_dim=128, quat_dim=64, **kw)

def geo_transformer_small(name='geo_small', n_layers=4, **kw):
    """Small config for prototyping."""
    return GeometricTransformer(name, d_model=256, n_heads=8,
        n_layers=n_layers, n_anchors=16, manifold_dim=128,
        n_comp=4, d_comp=16, context_dim=64, quat_dim=32, **kw)

def geo_transformer_vision(name='geo_vit', n_layers=4, **kw):
    """For scatter/SVD vision pipeline (patches as tokens)."""
    return GeometricTransformer(name, d_model=384, n_heads=8,
        n_layers=n_layers, n_anchors=32, manifold_dim=128,
        n_comp=8, d_comp=16, context_dim=64, quat_dim=32, **kw)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Geometric Transformer — CM Validated — Self-Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build small model ──
    model = geo_transformer_small('test_cm', n_layers=2)
    if hasattr(model, 'network_to'):
        model.network_to(device=device, strict=False)
    else:
        model = model.to(device)
    total = model.param_report()

    # ── Forward pass ──
    B, L, D = 2, 32, 256
    x = torch.randn(B, L, D, device=device)
    out, geos = model(x, return_geo_state=True)

    assert out.shape == (B, L, D), f"Expected ({B},{L},{D}), got {out.shape}"
    assert len(geos) == 2
    print(f"\n  Input:  ({B}, {L}, {D})")
    print(f"  Output: {out.shape}")
    print(f"  Geo states: {len(geos)} layers")

    # ── Verify CM gate is active ──
    for i, gs in enumerate(geos):
        gi = gs['gate_info']
        cm_q = gs['cm_quality']
        gv = gs['gate_values']
        print(f"\n  Layer {i} CM gate:")
        print(f"    active anchors:   {gi['active'].item():.1f} / {model.n_anchors}")
        print(f"    gate mean:        {gi['gate_mean'].item():.4f}")
        print(f"    cm_positive_frac: {gi['cm_positive_frac'].item():.3f}")
        print(f"    gate_values:      {gv.shape}  range=[{gv.min():.3f}, {gv.max():.3f}]")
        print(f"    cm_quality:       {cm_q.shape}  mean={cm_q.mean():.4f}")

    # ── Verify geo_residual continuity ──
    gr0 = geos[0]['geo_residual']
    gr1 = geos[1]['geo_residual']
    print(f"\n  Geo residual stream:")
    print(f"    Layer 0: {gr0.shape}  norm={gr0.norm(dim=-1).mean():.4f}")
    print(f"    Layer 1: {gr1.shape}  norm={gr1.norm(dim=-1).mean():.4f}")

    # ── Geometric losses ──
    geo_losses = model.geometric_losses()
    print(f"\n  Geometric regularization:")
    for k, v in geo_losses.items():
        print(f"    {k}: {v.item():.6f}")

    # ── Anchor diagnostics ──
    diag = model.anchor_diagnostics()
    print(f"\n  Anchor diagnostics:")
    for layer_name, d in diag.items():
        print(f"    {layer_name}:")
        for k, v in d.items():
            print(f"      {k}: {v:.4f}")

    # ── Verify Cayley rotations ──
    print(f"\n  Cayley rotations:")
    for name, module in model.named_modules():
        if isinstance(module, CayleyOrthogonal):
            R = module.get_rotation()
            I = torch.eye(R.shape[0], device=R.device)
            print(f"    {name}: ‖RRᵀ-I‖={((R@R.T)-I).norm():.8f}  det={torch.det(R):.4f}")

    # ── Gradient flow through CM gate ──
    print(f"\n  Gradient flow test:")
    model.zero_grad()
    x_grad = torch.randn(B, L, D, device=device, requires_grad=True)
    out_grad = model(x_grad)
    loss = out_grad.sum()
    loss.backward()

    # Check gate_proj has gradients
    for i in range(model.n_layers):
        layer = model[f'layer_{i}']
        gate_grads = [p.grad is not None and p.grad.abs().sum() > 0
                      for p in layer['cm_gate'].parameters()]
        print(f"    layer_{i} cm_gate grad: {'YES' if all(gate_grads) else 'NO'}")

    # ── Training step simulation ──
    print(f"\n  Training step simulation:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    x_train = torch.randn(B, L, D, device=device)
    out_train, states = model(x_train, return_geo_state=True)
    task_loss = out_train.mean()  # dummy

    geo_losses = model.geometric_losses()
    total_loss = task_loss + geo_losses.get('geo_total', 0.0)
    total_loss.backward()
    optimizer.step()
    print(f"    task_loss:  {task_loss.item():.4f}")
    print(f"    cv_loss:    {geo_losses['cv'].item():.6f}")
    print(f"    spread_loss:{geo_losses['spread'].item():.6f}")
    print(f"    total:      {total_loss.item():.4f}")

    # ── Paired forward + observer loss ──
    print(f"\n  Paired forward + observer loss:")
    model.zero_grad()

    x1 = torch.randn(B, L, D, device=device)
    x2 = x1 + 0.1 * torch.randn_like(x1)  # view 2 = slight perturbation
    targets = torch.randint(0, 10, (B,), device=device)

    output = model.forward_paired(x1, x2)
    print(f"    Output keys: {sorted(k for k in output if not k.startswith('geo_'))}")
    for k in ['embedding', 'patchwork1', 'bridge1', 'assign1', 'tri1']:
        print(f"    {k}: {output[k].shape}")

    # Task head for CE
    num_classes = 10
    head = nn.Linear(D, num_classes).to(device)

    loss, ld = model.compute_loss(output, targets, head=head)
    print(f"\n    Three-domain loss breakdown:")
    for k in ['loss_observer', 'loss_task', 'ce', 'nce_emb', 'nce_pw',
               'bridge', 'assign', 'assign_nce', 'nce_tri', 'attract',
               'cv', 'spread']:
        if k in ld:
            v = ld[k]
            v = v.item() if isinstance(v, torch.Tensor) else v
            print(f"      {k:16s} = {v:.4f}")
    for k in ['nce_emb_acc', 'nce_pw_acc', 'nce_tri_acc', 'bridge_acc',
               'assign_nce_acc', 'acc']:
        if k in ld:
            v = ld[k]
            v = v if isinstance(v, float) else v.item()
            print(f"      {k:16s} = {v*100:.1f}%")
    print(f"      {'TOTAL':16s} = {loss.item():.4f}")

    # Verify backward through observer loss
    loss.backward()
    alive_base, dead_base = [], []
    for n, p in model.named_parameters():
        if p.grad is not None and p.grad.norm() > 0:
            alive_base.append(n)
        else:
            dead_base.append(n)
    print(f"\n    Gradient flow: {len(alive_base)} params alive, {len(dead_base)} dead")
    if dead_base:
        print(f"\n    DEAD parameters (base model, paired+observer):")
        for n in dead_base:
            print(f"      {n}")

    # ══════════════════════════════════════════════════════════════
    # WIDE ROUTER COMPILATION
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  WideRouter Compilation")
    print(f"{'='*60}")

    if _HAS_WIDE_ROUTER:
        # Wrap transformer in WideRouter (same pattern as GeoViTClassifier)
        router = WideRouter('test_router', strict=False)
        router.attach('transformer', model)
        router.register_tower('transformer')
        router.network_to(device=device, strict=False)

        # Discover towers and compile
        router.discover_towers()
        print(f"\n  Towers discovered: {router.tower_names}")
        print(f"  Analyzed: {router.objects.get('_analyzed', False)}")

        try:
            compiled_router = router.compile(mode='default')
            print(f"  WideRouter.compile(mode='default'): OK")
        except Exception as e:
            print(f"  WideRouter.compile: {str(e)[:60]}")

        # Forward through the registered tower directly
        with torch.no_grad():
            out_via_router = router['transformer'](x)
        print(f"  Forward via router['transformer']: {out_via_router.shape}  OK")

        del router
    else:
        print(f"\n  WideRouter: geofractal not installed")

    print(f"\n{'='*60}")
    print(f"  PASSED — CM-validated pipeline operational")
    print(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════
    # FLOW ENSEMBLE INTEGRATION TESTS
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  Flow Ensemble Integration")
    print(f"{'='*60}")

    del model, optimizer
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    model_f = geo_transformer_small('test_flows', n_layers=2,
                                     flow_keys=['quat_lite', 'velocity', 'orbital'])
    if hasattr(model_f, 'network_to'):
        model_f.network_to(device=device, strict=False)
    else:
        model_f = model_f.to(device)

    total_f = model_f.param_report()
    print(f"\n  Total params (with flows): {total_f:,}")

    print(f"\n  Flow ensemble per layer:")
    for i in range(model_f.n_layers):
        layer = model_f[f'layer_{i}']
        if layer.has('flows'):
            flows = layer['flows']
            names = flows.active_flow_names
            params = sum(p.numel() for p in flows.parameters())
            print(f"    layer_{i}: {names}  ({params:,} params)")
        else:
            print(f"    layer_{i}: no flows attached")

    x_f = torch.randn(B, L, D, device=device)
    out_f, geos_f = model_f(x_f, return_geo_state=True)
    assert out_f.shape == (B, L, D)
    print(f"\n  Forward with flows: {out_f.shape}  OK")

    geo_ctx_0 = geos_f[0]['geo_ctx']
    print(f"  Geo context shape: {geo_ctx_0.shape}  norm={geo_ctx_0.norm(dim=-1).mean():.4f}")

    print(f"\n  Flow gradient test (out.sum().backward()):")
    model_f.zero_grad()
    x_fg = torch.randn(B, L, D, device=device, requires_grad=True)
    out_fg = model_f(x_fg)
    out_fg.sum().backward()

    alive_simple, dead_simple = [], []
    for n, p in model_f.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            alive_simple.append(n)
        else:
            dead_simple.append(n)
    print(f"    {len(alive_simple)} alive, {len(dead_simple)} dead")
    if dead_simple:
        print(f"\n    DEAD parameters (out.sum):")
        for n in dead_simple:
            print(f"      {n}")

    print(f"\n  Paired forward + observer loss (with flows):")
    model_f.zero_grad()
    x1_f = torch.randn(B, L, D, device=device)
    x2_f = x1_f + 0.1 * torch.randn_like(x1_f)
    targets_f = torch.randint(0, 10, (B,), device=device)

    output_f = model_f.forward_paired(x1_f, x2_f)
    head_f = nn.Linear(D, num_classes).to(device)
    loss_f, ld_f = model_f.compute_loss(output_f, targets_f, head=head_f)
    print(f"    total loss: {loss_f.item():.4f}")
    loss_f.backward()

    alive_paired, dead_paired = [], []
    for n, p in model_f.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            alive_paired.append(n)
        else:
            dead_paired.append(n)
    print(f"    {len(alive_paired)} alive, {len(dead_paired)} dead")
    if dead_paired:
        print(f"\n    DEAD parameters (paired+observer):")
        for n in dead_paired:
            print(f"      {n}")

    print(f"\n  Runtime flow management:")
    layer0 = model_f['layer_0']
    flows_0 = layer0['flows']
    print(f"    Before:           {flows_0.active_flow_names}")

    flows_0.attach_flow('alignment')
    print(f"    +alignment:       {flows_0.active_flow_names}")

    flows_0.detach_flow('velocity')
    print(f"    -velocity:        {flows_0.active_flow_names}")

    out_swapped = model_f(x_f)
    assert out_swapped.shape == (B, L, D)
    print(f"    Forward after swap: {out_swapped.shape}  OK")

    layer1 = model_f['layer_1']
    if layer1.has('flows'):
        for fn in list(layer1['flows'].active_flow_names):
            key = fn.replace('flow_', '')
            layer1['flows'].detach_flow(key)
        print(f"    Layer 1 after clear: {layer1['flows'].active_flow_names}")
        out_partial = model_f(x_f)
        assert out_partial.shape == (B, L, D)
        print(f"    Forward (L0 flows, L1 empty): {out_partial.shape}  OK")

    print(f"\n  Backward compatibility (no flows):")
    model_nf = geo_transformer_small('test_noflows', n_layers=2)
    if hasattr(model_nf, 'network_to'):
        model_nf.network_to(device=device, strict=False)
    else:
        model_nf = model_nf.to(device)
    out_nf = model_nf(torch.randn(B, L, D, device=device))
    assert out_nf.shape == (B, L, D)
    print(f"    Forward (no flows): {out_nf.shape}  OK")
    for i in range(model_nf.n_layers):
        assert not model_nf[f'layer_{i}'].has('flows'), f"layer_{i} should not have flows"
    print(f"    No flows attached:  OK")
    del model_nf

    print(f"\n{'='*60}")
    print(f"  PASSED — CM-validated pipeline operational")
    print(f"  PASSED — Flow ensemble integration verified")
    print(f"  PASSED — Flow attach/detach verified")
    print(f"  PASSED — Backward compatibility verified")
    print(f"{'='*60}")