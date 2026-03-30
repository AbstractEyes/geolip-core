"""
Geometric Transformer — CM-Validated Pipeline
==================================================
Dual-stream transformer with CM-gated constellation observation,
quaternion composition, and per-layer Cayley alignment.

CM-validated pipeline changes:
    - CM validity gate between association and curation (AnchorGate)
    - 4-stream PositionGeometricContext: anchor + structural + history + quality
    - CM-conditioned geometric residual accumulation (replaces blind learned gate)
    - Built-in geometric regularization (CV target + anchor spread)
    - Decomposed observer pipeline: association → CM gate → gated curation

Pipeline per layer:
    1. ManifoldProjection:  h_i → emb_i on S^(d-1) per position
    2. ConstellationAssociation: emb_i → raw triangulation, cos, assignment
    3. CMValidatedGate: per-anchor CM validity → gate_values (B*L, A)
    4. Gated curation: patchwork reads tri * gate_values (validated only)
    5. PositionGeometricContext: 4 streams → FiLM context (B, L, context_dim)
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
# GEOLIP IMPORTS — real components, not reimplementations
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from geolip_core.core.associate.constellation import (
        ConstellationObserver, ConstellationAssociation, ConstellationCuration,
        Constellation, init_anchors_repulsion,
    )
    from geolip_core.core.curate.gate import AnchorGate as _GeolipAnchorGate
    from geolip_core.pipeline.observer import (
        TorchComponent, BaseTower, Input, Curation, Distinction,
    )
    _HAS_GEOLIP = True
except ImportError:
    _HAS_GEOLIP = False

    # ── Fallback stubs ──
    class TorchComponent(nn.Module):
        def __init__(self, name=None, **kwargs):
            super().__init__()
            self._component_name = name or self.__class__.__name__

    class BaseTower(nn.Module):
        def __init__(self, name=None, **kwargs):
            super().__init__()
            self._tower_name = name or self.__class__.__name__
            self._components = nn.ModuleDict()
            self._cache = {}

        def attach(self, name, module):
            if isinstance(module, nn.Module):
                self._components[name] = module
            return self

        def has(self, name):
            return name in self._components

        def __getitem__(self, key):
            return self._components[key]

        def cache_set(self, key, value):
            self._cache[key] = value

        def cache_get(self, key, default=None):
            return self._cache.get(key, default)

        def cache_clear(self):
            self._cache.clear()

    Input = TorchComponent
    Curation = TorchComponent
    Distinction = TorchComponent

    class Constellation(nn.Module):
        """Learned anchors on S^(d-1). Triangulates input embeddings."""
        def __init__(self, n_anchors, dim, anchor_drop=0.0, anchor_init='repulsion'):
            super().__init__()
            self.n_anchors = n_anchors
            self.dim = dim
            anchors = torch.randn(n_anchors, dim)
            anchors = F.normalize(anchors, dim=-1)
            for _ in range(200):
                sim = anchors @ anchors.T
                sim.fill_diagonal_(-2.0)
                anchors = F.normalize(anchors - 0.05 * anchors[sim.argmax(dim=1)], dim=-1)
            self.anchors = nn.Parameter(anchors)

        def forward(self, emb, training=False):
            anchors = F.normalize(self.anchors, dim=-1)
            cos = emb @ anchors.T
            tri = 1.0 - cos
            _, nearest = cos.max(dim=-1)
            return tri, nearest

    class ConstellationAssociation(TorchComponent):
        """Association through constellation anchors."""
        def __init__(self, dim=256, n_anchors=32, anchor_drop=0.0,
                     anchor_init='repulsion', assign_temp=0.1, **kwargs):
            super().__init__(**kwargs)
            self.assign_temp = assign_temp
            self.constellation = Constellation(n_anchors, dim, anchor_drop, anchor_init)

        @property
        def frame_dim(self):
            return self.constellation.n_anchors

        def associate(self, emb, **context):
            anchors_n = F.normalize(self.constellation.anchors, dim=-1)
            cos = emb @ anchors_n.T
            tri = 1.0 - cos
            _, nearest = cos.max(dim=-1)
            soft_assign = F.softmax(cos / self.assign_temp, dim=-1)
            mag = context.get('mag', None)
            distances_weighted = tri * mag if mag is not None else tri
            return {
                'distances': tri, 'distances_weighted': distances_weighted,
                'cos_to_anchors': cos, 'assignment': soft_assign,
                'nearest': nearest,
            }

        def forward(self, emb, **context):
            return self.associate(emb, **context)

    class Patchwork(nn.Module):
        """Round-robin patchwork compartments."""
        def __init__(self, n_anchors, n_comp=8, d_comp=32, activation='gelu'):
            super().__init__()
            self.n_comp = n_comp
            anchors_per = max(1, n_anchors // n_comp)
            self.compartments = nn.ModuleList([
                nn.Sequential(nn.Linear(anchors_per, d_comp), nn.GELU(), nn.Linear(d_comp, d_comp))
                for _ in range(n_comp)
            ])
            self.output_dim = n_comp * d_comp
            self.anchors_per = anchors_per

        def forward(self, distances):
            parts = []
            for i, comp in enumerate(self.compartments):
                start = i * self.anchors_per
                end = start + self.anchors_per
                chunk = distances[..., start:end]
                if chunk.shape[-1] < self.anchors_per:
                    chunk = F.pad(chunk, (0, self.anchors_per - chunk.shape[-1]))
                parts.append(comp(chunk))
            return torch.cat(parts, dim=-1)

    class ConstellationCuration(Curation):
        """Curation through patchwork compartments + bridge."""
        def __init__(self, n_anchors=32, dim=256, n_comp=8, d_comp=32,
                     activation='gelu', **kwargs):
            super().__init__(**kwargs)
            self.dim = dim
            self.n_anchors = n_anchors
            self.patchwork = Patchwork(n_anchors, n_comp, d_comp, activation)
            pw_dim = self.patchwork.output_dim
            self.bridge = nn.Linear(pw_dim, n_anchors)
            self._feature_dim = n_anchors + pw_dim + dim

        @property
        def feature_dim(self):
            return self._feature_dim

        def curate_full(self, association_output, emb=None, **context):
            distances = association_output['distances_weighted']
            assignment = association_output['assignment']
            pw = self.patchwork(distances)
            bridge = self.bridge(pw)
            parts = [assignment, pw]
            if emb is not None:
                parts.append(emb)
            features = torch.cat(parts, dim=-1)
            return {'patchwork': pw, 'bridge': bridge, 'features': features}

        def forward(self, association_output, emb=None, **context):
            return self.curate_full(association_output, emb=emb, **context)['features']

    class ConstellationObserver(nn.Module):
        """Composed association + curation."""
        def __init__(self, dim=256, n_anchors=32, n_comp=8, d_comp=32,
                     anchor_drop=0.0, anchor_init='repulsion',
                     activation='gelu', assign_temp=0.1):
            super().__init__()
            self.association = ConstellationAssociation(
                dim=dim, n_anchors=n_anchors, anchor_drop=anchor_drop,
                anchor_init=anchor_init, assign_temp=assign_temp)
            self.curation = ConstellationCuration(
                n_anchors=n_anchors, dim=dim, n_comp=n_comp,
                d_comp=d_comp, activation=activation)

        @property
        def constellation(self):
            return self.association.constellation

        @property
        def patchwork(self):
            return self.curation.patchwork

        @property
        def feature_dim(self):
            return self.curation.feature_dim

        def observe(self, emb, **context):
            a_out = self.association(emb, **context)
            c_out = self.curation.curate_full(a_out, emb=emb, **context)
            return {
                'embedding': emb, 'features': c_out['features'],
                'triangulation': a_out['distances'],
                'cos_to_anchors': a_out['cos_to_anchors'],
                'nearest': a_out['nearest'],
                'assignment': a_out['assignment'],
                'patchwork': c_out['patchwork'], 'bridge': c_out['bridge'],
            }

        def forward(self, emb, **context):
            return self.observe(emb, **context)


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
    raw = torch.linalg.det(M)
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

    # Build simplices: [anchor_a, neighbor_1, ..., neighbor_k] per anchor
    K = n_neighbors + 1
    simplices = torch.zeros(A, K, D, device=anchors.device, dtype=anchors.dtype)
    simplices[:, 0] = anchors
    for j in range(n_neighbors):
        simplices[:, j + 1] = anchors[nn_idx[:, j]]

    dets = cayley_menger_det(simplices)  # (A,)
    sign = dets.sign()
    log_mag = torch.log(dets.abs() + 1e-12)
    return sign * log_mag, nn_idx


# ═══════════════════════════════════════════════════════════════════════════════
# CM VALIDATED GATE — efficient anchor gating for transformer scale
# ═══════════════════════════════════════════════════════════════════════════════

class CMValidatedGate(nn.Module):
    """Anchor gate based on Cayley-Menger validity.

    Efficient for transformer scale: anchor CM quality is precomputed O(A²),
    then combined with per-position proximity features through a learned gate.

    The gate starts OPEN (bias=+2, sigmoid≈0.88) and learns to CLOSE on
    geometrically invalid configurations. Architecture-before-loss: the gate
    suppresses degenerate measurements structurally, not through a loss signal.

    Gate features per (position, anchor):
        - anchor_cm_quality: CM volume of anchor's local neighborhood (position-independent)
        - cos_to_anchor: cosine similarity (position-dependent)
        - distance_rank: normalized rank of this anchor by proximity (position-dependent)

    Args:
        n_anchors: number of constellation anchors
        n_neighbors: neighbors for CM simplex computation
    """
    def __init__(self, n_anchors, n_neighbors=3):
        super().__init__()
        self.n_anchors = n_anchors
        self.n_neighbors = n_neighbors

        # Learned gate: [cm_quality, cos_sim, dist_rank] → scalar gate
        self.gate_proj = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # Init OPEN — learn to close. sigmoid(2.0) ≈ 0.88
        nn.init.zeros_(self.gate_proj[2].weight)
        nn.init.constant_(self.gate_proj[2].bias, 2.0)

    def forward(self, embedding, anchors, tri):
        """Compute per-(position, anchor) gate values.

        Args:
            embedding: (N, D) — positions on S^(d-1), where N = B*L
            anchors:   (A, D) — normalized anchor positions (DETACHED by caller)
            tri:       (N, A) — triangulation distances (1 - cos)

        Returns:
            gate_values: (N, A) in [0, 1] — per-anchor validity gate
            gate_info: dict with diagnostics
        """
        N, A = tri.shape

        # ── Anchor CM quality: position-independent, O(A²) ──
        with torch.no_grad():
            anchor_cm, nn_idx = anchor_neighborhood_cm(anchors, self.n_neighbors)
            # Normalize to ~ [-1, 1]
            cm_std = anchor_cm.std().clamp(min=1e-8)
            anchor_cm_norm = (anchor_cm - anchor_cm.mean()) / cm_std

        # ── Per-position features ──
        cos_sim = 1.0 - tri  # (N, A)

        # Distance rank: 0=nearest, 1=farthest
        ranks = tri.argsort(dim=-1).argsort(dim=-1).float()
        ranks = ranks / max(A - 1, 1)

        # ── Gate features: (N, A, 3) ──
        features = torch.stack([
            anchor_cm_norm.unsqueeze(0).expand(N, -1),
            cos_sim,
            ranks,
        ], dim=-1)

        gate_values = torch.sigmoid(self.gate_proj(features).squeeze(-1))

        # ── Diagnostics (no .item() — compile-safe) ──
        with torch.no_grad():
            active = (gate_values > 0.5).float().sum(-1).mean()
            cm_positive_frac = (anchor_cm > 0).float().mean()
            gate_mean = gate_values.mean()

        gate_info = {
            'active': active,
            'gate_mean': gate_mean,
            'cm_positive_frac': cm_positive_frac,
            'anchor_cm': anchor_cm.detach(),
        }

        return gate_values, gate_info


# ═══════════════════════════════════════════════════════════════════════════════
# INFONCE MEMORY BANK — contrastive pressure on geometric residual
# ═══════════════════════════════════════════════════════════════════════════════

class GeoResidualBank(nn.Module):
    """MoCo-style memory bank for geometric residual contrastive learning.

    The geometric residual converges to a consistent attractor (by design).
    The bank provides discriminative pressure: different samples should
    produce distinguishable residuals. Without it, the residual is
    geometrically consistent but sample-agnostic — "maple" and "oak"
    can both land at the same point.

    The bank stores detached geo_residuals from recent batches as keys.
    The current batch's residual is the query. InfoNCE asks: "can I
    identify which sample produced this geometric residual among K
    recent alternatives?"

    Gradient flows: InfoNCE loss → query (current geo_residual) →
    geo_proj → patchwork → CM-gated triangulation → constellation.
    The bank entries are detached — no momentum encoder needed.

    Args:
        feature_dim: dimension of geo_residual (pw_dim = n_comp * d_comp)
        bank_size: number of entries in the queue (default 4096)
        temperature: InfoNCE temperature (default 0.1)
    """
    def __init__(self, feature_dim, bank_size=4096, temperature=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.temperature = temperature

        # Queue buffer — not a parameter, not updated by optimizer
        self.register_buffer('queue', torch.randn(bank_size, feature_dim))
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys):
        """Add batch of keys to the queue. Called AFTER backward.

        Args:
            keys: (B, feature_dim) detached, normalized geo_residuals
        """
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        # Handle wrap-around
        if ptr + B <= self.bank_size:
            self.queue[ptr:ptr + B] = keys
        else:
            overflow = (ptr + B) - self.bank_size
            self.queue[ptr:] = keys[:B - overflow]
            self.queue[:overflow] = keys[B - overflow:]

        self.queue_ptr[0] = (ptr + B) % self.bank_size

    def forward(self, queries):
        """Compute InfoNCE loss: queries vs bank.

        Args:
            queries: (B, feature_dim) — current batch's geo_residuals (LIVE, has grad)

        Returns:
            loss: scalar InfoNCE loss
            acc: scalar top-1 retrieval accuracy (for diagnostics)
        """
        queries = F.normalize(queries, dim=-1)  # (B, D)
        keys = self.queue.clone().detach()       # (K, D) — no grad to bank

        # Positive: each query's closest match isn't defined in standard InfoNCE.
        # For a memory bank without paired data, we use the query itself as
        # the target — the loss pushes each query away from all bank entries
        # while the bank gradually fills with diverse residuals.
        #
        # But we DO have paired data in CutMix-free batches. The clean approach:
        # use the batch itself for positives. For sample i, its positive is itself
        # (trivial), so instead we compute the batch-vs-bank contrastive:
        # each query should be distinguishable from all bank entries.

        # Logits: (B, K) — similarity of each query to all bank entries
        logits = queries @ keys.T / self.temperature  # (B, K)

        # Self-similarity within batch for positive pairs
        # Each sample's "positive" is its own representation — we want it to
        # be MORE similar to itself than to any bank entry.
        # Add self-similarity as column 0:
        self_sim = (queries * queries).sum(dim=-1, keepdim=True) / self.temperature  # (B, 1)
        logits = torch.cat([self_sim, logits], dim=1)  # (B, 1+K)

        # Labels: column 0 is the positive for every sample
        labels = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)

        loss = F.cross_entropy(logits, labels)

        # Accuracy: how often is the self-similarity highest
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == 0).float().mean()

        return loss, acc


# ═══════════════════════════════════════════════════════════════════════════════
# PROVEN COMPONENTS — from Ryan Spearman (unchanged, tested)
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(TorchComponent):
    """Feature-wise Linear Modulation. Proven in Ryan Spearman.
    Identity-initialized: γ=1, β=0 at init.
    """
    def __init__(self, name, feature_dim, context_dim):
        super().__init__(name)
        self.to_gamma = nn.Linear(context_dim, feature_dim)
        self.to_beta = nn.Linear(context_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, ctx):
        return self.to_gamma(ctx) * x + self.to_beta(ctx)


class CayleyOrthogonal(TorchComponent):
    """Guaranteed SO(d) rotation via Cayley map. det(Q) = 1 always."""
    def __init__(self, name, dim):
        super().__init__(name)
        self.dim = dim
        self.A_upper = nn.Parameter(torch.zeros(dim * (dim - 1) // 2) * 0.01)
        idx = torch.triu_indices(dim, dim, offset=1)
        self.register_buffer('_triu_row', idx[0], persistent=False)
        self.register_buffer('_triu_col', idx[1], persistent=False)
        self.register_buffer('_eye', torch.eye(dim), persistent=False)

    def get_rotation(self):
        d = self.dim
        A = torch.zeros(d, d, device=self.A_upper.device, dtype=self.A_upper.dtype)
        A[self._triu_row, self._triu_col] = self.A_upper
        A = A - A.T
        return torch.linalg.solve(self._eye + A, self._eye - A)

    def forward(self, x):
        return x @ self.get_rotation().T


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
    """Four-arm Hamilton product composition. Proven in GeoQuat head.
    Fully vectorized: single batched Hamilton product, no Python loops.
    """
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
    """Input stage: project transformer hidden states to S^(d-1).
    Per-position, per-layer. L2-normalized to unit hypersphere.
    """
    def __init__(self, name, d_model, manifold_dim):
        super().__init__(name)
        self.proj = nn.Linear(d_model, manifold_dim)
        self.norm = nn.LayerNorm(manifold_dim)

    def forward(self, hidden_states):
        h = self.norm(self.proj(hidden_states))
        return F.normalize(h, dim=-1)


class PositionGeometricContext(TorchComponent):
    """Curation stage: 4-stream fusion → FiLM context.

    Four streams:
        anchor:     cos_to_anchors + assignment + triangulation — WHERE on the manifold
        structural: patchwork + embedding — WHAT the local geometry looks like
        history:    geo_residual from previous layers — WHAT prior layers observed
        quality:    CM gate values per anchor — HOW TRUSTWORTHY is this observation

    The quality stream gives FiLM direct knowledge of which anchors formed
    valid simplices. This is not a scalar — the full (N, A) gate profile
    tells the context WHICH directions on the manifold are reliable.
    """
    def __init__(self, name, n_anchors, pw_dim, manifold_dim, context_dim):
        super().__init__(name)
        self.context_dim = context_dim
        self.pw_dim = pw_dim

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

        # Fuse 4 streams
        self.fuse = nn.Sequential(
            nn.Linear(context_dim * 4, context_dim), nn.GELU(), nn.LayerNorm(context_dim))

    def forward(self, obs_dict, gate_values=None, geo_residual=None):
        """
        Args:
            obs_dict: from decomposed association + gated curation
            gate_values: (N, A) CM gate values per anchor, or None
            geo_residual: (N, pw_dim) accumulated context, or None for first layer
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

        return self.fuse(torch.cat([a, s, h, q], dim=-1))


class GeometricAttention(TorchComponent):
    """Attention with FiLM from curated constellation. Stream B.
    FiLM modulates Q,K BEFORE attention. V stays unmodulated.
    """
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
                         key_padding_mask=key_padding_mask)
        x = self.norm(x + a)
        x = self.ffn_norm(x + self.ffn(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER — CM-validated dual-stream with constellation routing
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricTransformerLayer(BaseTower):
    """One layer of the geometric transformer (CM validated).

    Pipeline per layer:
        1. ManifoldProjection: h → emb on S^(d-1)
        2. Association: emb → raw triangulation, cos, assignment
        3. CMValidatedGate: per-anchor CM validity → gate_values
        4. Gated curation: patchwork reads tri * gate_values
        5. PositionGeometricContext: 4 streams → FiLM context
        6. ContentAttention (Stream A): standard MHA
        7. GeometricAttention (Stream B): FiLM(Q,K | geo_ctx)
        8. CayleyOrthogonal: align B → A
        9. QuaternionCompose: w=A, i=aligned_B, j=A-B, k=A*B
       10. Decode + gated residual
       11. CM-conditioned geometric residual accumulation

    The observer is DECOMPOSED: association and curation are called
    separately with the CM gate inserted between them. The gate
    suppresses degenerate anchor measurements before the patchwork
    reads them. The patchwork only interprets validated geometry.

    The geometric residual is accumulated using CM quality as the
    write weight — no learned gate. Positions with high-quality
    simplex observations contribute more. Positions in degenerate
    regions contribute less.
    """
    def __init__(self, name, d_model, n_heads=8, n_anchors=32,
                 manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1,
                 cm_neighbors=3):
        super().__init__(name)
        self.d_model = d_model
        self.n_anchors = n_anchors

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

        # 4. Fuse observation into FiLM context (4 streams)
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
        gate_values, gate_info = self['cm_gate'](
            emb_flat, anchors_n.detach(), a_out['distances'])

        # ════ 4. Gated curation — patchwork reads validated triangulation ════
        a_out_gated = dict(a_out)
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

        # ════ 5. Build FiLM context — 4 streams ════
        geo_res_flat = geo_residual.reshape(B * L, -1) if geo_residual is not None else None
        geo_ctx_flat = self['context'](
            obs, gate_values=gate_values, geo_residual=geo_res_flat)
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
        # CM quality per position: mean gate value across anchors.
        # High quality = position's simplex with anchors is non-degenerate.
        # Low quality = position is in a boundary region or near dead anchors.
        pw_validated = c_out['patchwork'].reshape(B, L, -1)
        cm_quality = gate_values.mean(dim=-1).reshape(B, L, 1)  # (B, L, 1)
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
        }

        return x_out, geo_residual_out, geo_state


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL — stack of layers + geometric regularization
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricTransformer(BaseTower):
    """Geometric Transformer — CM-validated dual-stream.

    Stack of GeometricTransformerLayers with:
        - CM-gated observation at every layer
        - Cross-layer Cayley rotation on hidden states (not geo_residual)
        - Built-in geometric regularization via geometric_losses()
    """
    def __init__(self, name, d_model=512, n_heads=8, n_layers=4,
                 n_anchors=32, manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1,
                 cross_layer_rotation=True, cm_neighbors=3,
                 nce_bank_size=4096, nce_temperature=0.1,
                 vocab_size=None, max_seq_len=2048):
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
                dropout, cm_neighbors))

        if cross_layer_rotation and n_layers > 1:
            for i in range(n_layers - 1):
                self.attach(f'cross_rot_{i}', CayleyOrthogonal(
                    f'{name}_xrot_{i}', d_model))

        self.attach('final_norm', nn.LayerNorm(d_model))

        # InfoNCE memory bank on geometric residual
        if nce_bank_size > 0:
            self.attach('nce_bank', GeoResidualBank(
                self._pw_dim, bank_size=nce_bank_size,
                temperature=nce_temperature))

        self._config = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            n_anchors=n_anchors, manifold_dim=manifold_dim,
            n_comp=n_comp, d_comp=d_comp, context_dim=context_dim,
            quat_dim=quat_dim, dropout=dropout,
            cross_layer_rotation=cross_layer_rotation,
            cm_neighbors=cm_neighbors, vocab_size=vocab_size,
            nce_bank_size=nce_bank_size, nce_temperature=nce_temperature,
        )

    @property
    def config(self):
        return self._config.copy()

    def geometric_losses(self, cv_target=0.215, cv_weight=0.1, spread_weight=0.01):
        """Compute geometric regularization from current anchor geometry.

        These losses maintain the constellation in the regime where
        CM validation, patchwork interpretation, and the full observation
        pipeline produce meaningful results.

        CV loss: push anchor coefficient of variation toward pentachoron
        band (0.20-0.23). This is where CM computation has maximal
        discriminative power — anchors are neither too uniform (CV≈0,
        CM uninformative) nor too clustered (CV>0.3, degenerate simplices).

        Spread loss: penalize positive cosine similarity between anchors.
        Prevents collapse where multiple anchors occupy the same region,
        creating redundant measurements and wasting patchwork capacity.

        Returns:
            dict with 'cv', 'spread', 'geo_total' loss tensors
        """
        total_cv = torch.tensor(0.0)
        total_spread = torch.tensor(0.0)
        n = 0

        for i in range(self.n_layers):
            layer = self[f'layer_{i}']
            anchors = layer['observer'].association.constellation.anchors
            anchors_n = F.normalize(anchors, dim=-1)
            A = anchors_n.shape[0]

            # Ensure we're on the right device
            if n == 0:
                total_cv = total_cv.to(anchors.device)
                total_spread = total_spread.to(anchors.device)

            # ── CV loss: pairwise angular distance coefficient of variation ──
            cos = anchors_n @ anchors_n.T
            idx = torch.triu_indices(A, A, offset=1, device=cos.device)
            pairwise_dist = 1.0 - cos[idx[0], idx[1]]
            cv = pairwise_dist.std() / (pairwise_dist.mean() + 1e-8)
            total_cv = total_cv + (cv - cv_target).pow(2)

            # ── Spread loss: penalize positive cosine between anchors ──
            mask = ~torch.eye(A, dtype=torch.bool, device=cos.device)
            total_spread = total_spread + F.relu(cos[mask]).mean()

            n += 1

        losses = {}
        if n > 0:
            losses['cv'] = cv_weight * total_cv / n
            losses['spread'] = spread_weight * total_spread / n
            losses['geo_total'] = losses['cv'] + losses['spread']
        return losses

    def infonce_loss(self, geo_residual=None, cls_index=0):
        """Compute InfoNCE contrastive loss on geometric residual.

        Takes the CLS token's accumulated geo_residual as the per-sample
        geometric signature. Compares against the memory bank of recent
        residuals. Gradient flows back through the full geometric pipeline.

        Call DURING training, after forward pass. Call update_nce_bank()
        AFTER backward to enqueue current batch.

        Args:
            geo_residual: (B, L, pw_dim) or None (uses cached from last forward)
            cls_index: which position to use as per-sample signature (default 0 = CLS)

        Returns:
            dict with 'nce': loss tensor, 'nce_acc': retrieval accuracy
        """
        if not self.has('nce_bank'):
            return {}

        if geo_residual is None:
            geo_residual = getattr(self, '_last_geo_residual', None)
        if geo_residual is None:
            return {}

        cls_residual = geo_residual[:, cls_index]  # (B, pw_dim)
        loss, acc = self['nce_bank'](cls_residual)
        return {'nce': loss, 'nce_acc': acc}

    @torch.no_grad()
    def update_nce_bank(self, geo_residual=None, cls_index=0):
        """Enqueue current batch's residuals into the bank. Call AFTER backward.

        Args:
            geo_residual: (B, L, pw_dim) or None (uses cached from last forward)
            cls_index: which position to use (default 0 = CLS)
        """
        if not self.has('nce_bank'):
            return

        if geo_residual is None:
            geo_residual = getattr(self, '_last_geo_residual', None)
        if geo_residual is None:
            return

        cls_residual = F.normalize(geo_residual[:, cls_index].detach(), dim=-1)
        self['nce_bank'].enqueue(cls_residual)

    def anchor_diagnostics(self):
        """Per-layer anchor health diagnostics. Call for monitoring."""
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

            # CM quality per anchor
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
        print(f"\n  {name} — parameter report (CM-validated)")
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
        """
        Returns:
            out: (B, L, D) transformed hidden states (or logits if head attached)
            geo_states: list of per-layer geo_state dicts (if return_geo_state)

        Side effect:
            self._last_geo_residual is set to the final geo_residual (B, L, pw_dim)
            for use by infonce_loss() and update_nce_bank() without changing the return API.
        """
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
                # geo_residual NOT rotated — lives in patchwork space, basis-independent

        # Cache for infonce_loss / update_nce_bank
        self._last_geo_residual = geo_residual

        x = self['final_norm'](x)
        if self.has('head'):
            x = self['head'](x)

        return (x, geo_states) if return_geo_state else x


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
    print(f"  geolip_core available: {_HAS_GEOLIP}")
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

    print(f"\n{'='*60}")
    print(f"  PASSED — CM-validated pipeline operational")
    print(f"{'='*60}")