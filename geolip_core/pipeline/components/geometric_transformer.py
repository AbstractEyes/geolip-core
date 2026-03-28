"""
Geometric Transformer — GeoLIP Pipeline Integration
=====================================================
Dual-stream transformer with constellation-routed attention,
quaternion composition, and per-layer Cayley alignment.

Uses REAL geolip_core components:
    core.associate.constellation  — ConstellationObserver (anchors + triangulation + patchwork)
    core.curate.gate              — AnchorGate (CM determinant validity)
    core.align.procrustes         — CayleyOrthogonal rotation in SO(d)
    pipeline.observer             — TorchComponent / BaseTower interfaces

NEW components (transformer-specific):
    ManifoldProjection            — Input stage: hidden_state → S^(d-1)
    PositionGeometricContext      — Curation: constellation output → FiLM context
    FiLMLayer                     — Feature-wise Linear Modulation (proven in Ryan Spearman)
    GeometricAttention            — Attention with FiLM on Q,K from curated constellation
    QuaternionCompose             — Hamilton product of dual-stream outputs (proven)
    CayleyOrthogonal              — SO(d) rotation via Cayley map (proven)
    DualStreamBlock               — Content + geometric streams, aligned + composed
    GeometricTransformerLayer     — Full layer: project → observe → attend → compose
    GeometricTransformer          — Stack of layers with cross-layer rotation

Architecture per layer:
    1. ManifoldProjection:  h_i → emb_i on S^(d-1) per position
    2. ConstellationObserver: emb_i → {triangulation, assignment, patchwork, bridge}
    3. PositionGeometricContext: constellation output → (B, L, context_dim)
    4. Stream A (content):  standard self-attention
    5. Stream B (geometric): attention with FiLM(Q,K | geo_ctx), V unmodulated
    6. CayleyOrthogonal:    align B → A basis
    7. QuaternionCompose:   w=content, i=aligned_geo, j=disagree, k=agree
    8. Gated residual

Design principles from Ryan Spearman (ρ=0.309, 76/84 wins):
    - FiLM on Q,K ONLY — geometry routes attention, V stays pure
    - FiLM on individual arms BEFORE composition, not after
    - Quaternion algebra as structural regularizer (non-commutative coupling)
    - Disagreement arm (j) carries the transferable signal
    - CayleyOrthogonal guarantees pure rotation (det=1 always)
    - Never global average pool — per-position geometric context

Usage:
    from geometric_transformer import GeometricTransformer

    model = GeometricTransformer('geo_xfmr', d_model=512, n_layers=4)
    out = model(hidden_states)

    # Or as a head on frozen ESM-2:
    model = GeometricTransformer('esm2_geo', d_model=1280, n_layers=6)
    out = model(esm2_hidden_states)

Dependencies:
    pip install geolip-core  (includes constellation, patchwork, gate, observer interfaces)
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
    from geolip_core.core.curate.gate import AnchorGate
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
            self.anchor_drop = anchor_drop
            anchors = torch.randn(n_anchors, dim)
            # Repulsion-initialized
            anchors = F.normalize(anchors, dim=-1)
            for _ in range(200):
                sim = anchors @ anchors.T
                sim.fill_diagonal_(-2.0)
                anchors = F.normalize(anchors - 0.05 * anchors[sim.argmax(dim=1)], dim=-1)
            self.anchors = nn.Parameter(anchors)

        def triangulate(self, emb, training=False):
            anchors = F.normalize(self.anchors, dim=-1)
            cos = emb @ anchors.T
            tri = 1.0 - cos
            _, nearest = cos.max(dim=-1)
            return tri, nearest

        def forward(self, emb, training=False):
            return self.triangulate(emb, training)

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
# PROVEN COMPONENTS — from Ryan Spearman (unchanged, tested)
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(TorchComponent):
    """Feature-wise Linear Modulation. Proven in Ryan Spearman.

    Produces γ * x + β from geometric context.
    Identity-initialized: γ=1, β=0 at init.
    """
    def __init__(self, name, feature_dim, context_dim):
        super().__init__(name)
        self.to_gamma = nn.Linear(context_dim, feature_dim)
        self.to_beta = nn.Linear(context_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight); nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, ctx):
        """x: (B, L, D), ctx: (B, L, C) → (B, L, D)"""
        return self.to_gamma(ctx) * x + self.to_beta(ctx)


class CayleyOrthogonal(TorchComponent):
    """Guaranteed SO(d) rotation via Cayley map. Proven in Procrustes alignment.

    Q = (I - A)(I + A)^(-1) where A is skew-symmetric.
    det(Q) = 1 always. ‖R-I‖ ≈ 4.1 at convergence in SO(256).

    Uses the parent router's cache system for rotation storage.
    CompileRouter traces through cleanly — no Python-level guards.
    """
    def __init__(self, name, dim):
        super().__init__(name)
        self.dim = dim
        self.A_upper = nn.Parameter(torch.zeros(dim * (dim - 1) // 2) * 0.01)
        # Pre-compute index tensors as buffers for device tracking
        idx = torch.triu_indices(dim, dim, offset=1)
        self.register_buffer('_triu_row', idx[0], persistent=False)
        self.register_buffer('_triu_col', idx[1], persistent=False)
        self.register_buffer('_eye', torch.eye(dim), persistent=False)

    def _cache_key(self):
        return f'{self.name}_rotation'

    def compute_rotation(self):
        """Build SO(d) rotation from skew-symmetric parameters."""
        d = self.dim
        A = torch.zeros(d, d, device=self.A_upper.device, dtype=self.A_upper.dtype)
        A[self._triu_row, self._triu_col] = self.A_upper
        A = A - A.T
        return torch.linalg.solve(self._eye + A, self._eye - A)

    def get_rotation(self):
        """Get rotation, using parent router cache when available."""
        # Check parent cache (managed by router lifecycle)
        if hasattr(self, 'parent') and self.parent is not None:
            key = self._cache_key()
            cached = self.parent.cache_get(key)
            if cached is not None:
                return cached
            R = self.compute_rotation()
            self.parent.cache_set(key, R)
            return R
        return self.compute_rotation()

    def forward(self, x):
        """(..., dim) → (..., dim) rotated."""
        return x @ self.get_rotation().T


def quaternion_multiply(q1, q2):
    """Hamilton product. q = (w, x, y, z) along dim=-2.

    Supports batched: (..., 4, D) × (..., 4, D) → (..., 4, D)
    Or scalar:        (..., 4) × (..., 4) → (..., 4)
    """
    w1, x1, y1, z1 = q1.unbind(-2) if q1.dim() >= 2 and q1.shape[-2] == 4 else q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-2) if q2.dim() >= 2 and q2.shape[-2] == 4 else q2.unbind(-1)
    stack_dim = -2 if q1.dim() >= 2 and q1.shape[-2] == 4 else -1
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=stack_dim)


def quaternion_multiply_batched(q1, q2):
    """Hamilton product on (B, 4, D) tensors. Fully vectorized, no loops.

    Each of the 4 slices along dim=1 is one quaternion component.
    The D dimension is batched — all D quaternions multiplied in parallel.
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=1)  # (B, 4, D)


class QuaternionCompose(TorchComponent):
    """Four-arm Hamilton product composition. Proven in GeoQuat head.

    The algebra forces cross-term interactions between arms.
    Arms cannot independently memorize — the non-commutative
    product couples their outputs as structural regularizer.

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
        """Each arm: (B, L, D) → composed: (B, L, 4*quat_dim)"""
        shape = arm_w.shape[:-1]
        D = arm_w.shape[-1]
        flat = arm_w.dim() > 2
        if flat:
            arm_w = arm_w.reshape(-1, D); arm_i = arm_i.reshape(-1, D)
            arm_j = arm_j.reshape(-1, D); arm_k = arm_k.reshape(-1, D)

        # q: (N, 4, quat_dim) — stack 4 projected arms as quaternion components
        q = torch.stack([self.proj_w(arm_w), self.proj_i(arm_i),
                         self.proj_j(arm_j), self.proj_k(arm_k)], dim=1)
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)

        # r: (N, 4, quat_dim) — broadcast learned rotation
        r = self.rotation.expand(q.shape[0], -1, -1)
        r = r / (r.norm(dim=1, keepdim=True) + 1e-8)

        # Single batched Hamilton product over all quat_dim simultaneously
        # (N, 4, quat_dim) × (N, 4, quat_dim) → (N, 4, quat_dim)
        composed = quaternion_multiply_batched(r, q)

        # Flatten 4 × quat_dim → 4*quat_dim
        composed = composed.reshape(q.shape[0], -1)

        if flat:
            composed = composed.reshape(*shape, -1)
        return composed


# ═══════════════════════════════════════════════════════════════════════════════
# NEW COMPONENTS — transformer-specific, built for this architecture
# ═══════════════════════════════════════════════════════════════════════════════

class ManifoldProjection(TorchComponent):
    """Input stage: project transformer hidden states to S^(d-1).

    Per-position, per-layer projection from model space to the
    constellation's embedding space. L2-normalized to sit on the
    unit hypersphere.

    This is the tap — it reads the representation without modifying it.
    """
    def __init__(self, name, d_model, manifold_dim):
        super().__init__(name)
        self.proj = nn.Linear(d_model, manifold_dim)
        self.norm = nn.LayerNorm(manifold_dim)

    def forward(self, hidden_states):
        """(B, L, D) → (B, L, manifold_dim) on S^(manifold_dim - 1)"""
        h = self.norm(self.proj(hidden_states))
        return F.normalize(h, dim=-1)


class PositionGeometricContext(TorchComponent):
    """Curation stage: constellation observation → FiLM context vector.

    Takes the full observation dict from ConstellationObserver and fuses
    it into a per-position conditioning vector for FiLM layers.

    Processes: cos_to_anchors, assignment, patchwork, embedding.
    These are the same features the GeoQuat head used — validated on
    ProteinGym across 84 unseen proteins.
    """
    def __init__(self, name, n_anchors, pw_dim, manifold_dim, context_dim):
        super().__init__(name)
        # Anchor features: cos + assignment + triangulation = 3 * n_anchors
        self.anchor_mlp = nn.Sequential(
            nn.Linear(n_anchors * 3, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        # Structural features: patchwork + embedding
        self.struct_mlp = nn.Sequential(
            nn.Linear(pw_dim + manifold_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        # Fuse anchor + structural
        self.fuse = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )

    def forward(self, obs_dict):
        """
        Args:
            obs_dict: from ConstellationObserver.observe(), keys:
                cos_to_anchors: (B*L, A)
                assignment: (B*L, A)
                triangulation: (B*L, A)
                patchwork: (B*L, pw_dim)
                embedding: (B*L, manifold_dim)
        Returns:
            (B*L, context_dim) geometric context
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
        return self.fuse(torch.cat([a, s], dim=-1))


class GeometricAttention(TorchComponent):
    """Attention with FiLM from curated constellation. Stream B.

    FiLM modulates Q and K BEFORE attention — the constellation
    position controls WHERE attention flows. V stays unmodulated.
    FiLM between FFN layers conditions the nonlinearity.

    Proven principle: context before composition, not after.
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

        # FiLM on Q and K — geometry routes attention
        self.film_q = FiLMLayer(f'{name}_film_q', d_model, context_dim)
        self.film_k = FiLMLayer(f'{name}_film_k', d_model, context_dim)

        self.norm = nn.LayerNorm(d_model)

        # FFN with FiLM between layers
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.film_ffn = FiLMLayer(f'{name}_film_ffn', d_model * 4, context_dim)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, geo_ctx, attn_mask=None, key_padding_mask=None):
        """
        x: (B, L, D), geo_ctx: (B, L, C) → (B, L, D)
        """
        B, L, D = x.shape
        H, HD = self.n_heads, self.head_dim

        Q = self.film_q(self.w_q(x), geo_ctx)
        K = self.film_k(self.w_k(x), geo_ctx)
        V = self.w_v(x)  # V unmodulated — content stays pure

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

        # FFN with geometric FiLM between layers
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
# LAYER — dual-stream with constellation routing
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricTransformerLayer(BaseTower):
    """One layer of the geometric transformer.

    Pipeline per layer:
        1. ManifoldProjection: h_i → emb_i on S^(manifold_dim - 1)
        2. ConstellationObserver: emb_i → {triangulation, assignment, patchwork, ...}
        3. PositionGeometricContext: observation → FiLM context (B, L, context_dim)
        4. ContentAttention (Stream A): standard MHA
        5. GeometricAttention (Stream B): FiLM(Q,K | geo_ctx), V pure
        6. CayleyOrthogonal: align B basis → A basis
        7. QuaternionCompose: w=A, i=aligned_B, j=A-B, k=A*B
        8. Decode + gated residual

    Access:
        layer['projection']  → ManifoldProjection
        layer['observer']    → ConstellationObserver
        layer['context']     → PositionGeometricContext
        layer['content']     → ContentAttention
        layer['geometric']   → GeometricAttention
        layer['rotation']    → CayleyOrthogonal
        layer['compose']     → QuaternionCompose
    """
    def __init__(self, name, d_model, n_heads=8, n_anchors=32,
                 manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1):
        super().__init__(name)
        self.d_model = d_model

        # 1. Project to manifold
        self.attach('projection', ManifoldProjection(
            f'{name}_proj', d_model, manifold_dim))

        # 2. Constellation observer (real association + curation)
        self.attach('observer', ConstellationObserver(
            dim=manifold_dim, n_anchors=n_anchors,
            n_comp=n_comp, d_comp=d_comp))

        # 3. Fuse observation into FiLM context
        pw_dim = self['observer'].curation.patchwork.output_dim
        self.attach('context', PositionGeometricContext(
            f'{name}_ctx', n_anchors, pw_dim, manifold_dim, context_dim))

        # 4. Stream A: content
        self.attach('content', ContentAttention(
            f'{name}_content', d_model, n_heads, dropout))

        # 5. Stream B: geometric
        self.attach('geometric', GeometricAttention(
            f'{name}_geo', d_model, n_heads, context_dim, dropout))

        # 6. Cayley rotation: align B → A
        self.attach('rotation', CayleyOrthogonal(f'{name}_cayley', d_model))

        # 7. Quaternion composition
        self.attach('compose', QuaternionCompose(
            f'{name}_quat', d_model, quat_dim))

        # 8. Decode + gate
        self.attach('decode', nn.Sequential(
            nn.Linear(quat_dim * 4, d_model), nn.GELU(), nn.LayerNorm(d_model)))
        self.attach('gate', nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()))

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (B, L, D) input hidden states

        Returns:
            x_out: (B, L, D) transformed hidden states
            geo_state: dict with full geometric residual:
                'embedding':      (B, L, manifold_dim)  position on S^(d-1)
                'geo_ctx':        (B, L, context_dim)   compressed FiLM context
                'triangulation':  (B, L, A)             cosine distances to anchors
                'cos_to_anchors': (B, L, A)             raw cosine similarities
                'assignment':     (B, L, A)             soft assignment
                'nearest':        (B, L)                nearest anchor index
                'patchwork':      (B, L, pw_dim)        compartment features
                'bridge':         (B, L, A)             patchwork's assignment estimate
                'content':        (B, L, D)             Stream A output
                'geometric':      (B, L, D)             Stream B output (pre-rotation)
                'composed':       (B, L, 4*quat_dim)    raw quaternion composition
        """
        B, L, D = x.shape

        # 1. Project to manifold: per-position embedding on S^(d-1)
        emb = self['projection'](x)  # (B, L, manifold_dim)

        # 2. Constellation observation: flatten to (B*L, manifold_dim) for observer
        emb_flat = emb.reshape(B * L, -1)
        obs = self['observer'].observe(emb_flat)

        # 3. Build FiLM context
        geo_ctx_flat = self['context'](obs)  # (B*L, context_dim)
        geo_ctx = geo_ctx_flat.reshape(B, L, -1)  # (B, L, context_dim)

        # 4. Stream A: content attention
        a_out = self['content'](x, attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)

        # 5. Stream B: geometric attention
        b_out = self['geometric'](x, geo_ctx, attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask)

        # 6. Cayley rotation: align B → A
        b_aligned = self['rotation'](b_out)

        # 7. Quaternion composition
        #    w = content (what does standard attention think?)
        #    i = aligned geometry (what does geometric attention think?)
        #    j = disagreement (where do they diverge? — the surprise signal)
        #    k = agreement (where do they converge? — the confidence signal)
        composed = self['compose'](
            arm_w=a_out, arm_i=b_aligned,
            arm_j=a_out - b_aligned, arm_k=a_out * b_aligned)

        # 8. Decode + gated residual
        decoded = self['decode'](composed)
        g = self['gate'](torch.cat([x, decoded], dim=-1))
        x_out = g * decoded + (1 - g) * x

        # 9. Build full geometric state — reshape everything back to (B, L, ...)
        def unflatten(t):
            if t is None: return None
            if t.dim() == 1: return t.reshape(B, L)        # (B*L,) → (B, L)
            return t.reshape(B, L, *t.shape[1:])            # (B*L, ...) → (B, L, ...)

        geo_state = {
            'embedding':      emb,                          # already (B, L, manifold_dim)
            'geo_ctx':        geo_ctx,                      # already (B, L, context_dim)
            'triangulation':  unflatten(obs['triangulation']),
            'cos_to_anchors': unflatten(obs['cos_to_anchors']),
            'assignment':     unflatten(obs['assignment']),
            'nearest':        unflatten(obs['nearest']),
            'patchwork':      unflatten(obs['patchwork']),
            'bridge':         unflatten(obs['bridge']),
            'content':        a_out,                        # (B, L, D)
            'geometric':      b_out,                        # (B, L, D) pre-rotation
            'composed':       composed,                     # (B, L, 4*quat_dim)
        }

        return x_out, geo_state


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL — stack of layers
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricTransformer(BaseTower):
    """Geometric Transformer — dual-stream with constellation routing.

    Stack of GeometricTransformerLayers. Optional cross-layer Cayley
    rotation aligns each layer's output basis to the next layer's
    expected input.

    Access:
        model['layer_0']      → first layer
        model['cross_rot_0']  → cross-layer rotation 0→1
        model['final_norm']   → output normalization

    Args:
        name: tower identity
        d_model: transformer model dimension
        n_heads: attention heads per stream
        n_layers: number of geometric transformer layers
        n_anchors: constellation anchor points
        manifold_dim: dimension of S^(d-1) for constellation
        n_comp: patchwork compartments
        d_comp: hidden dim per compartment
        context_dim: FiLM conditioning dimension
        quat_dim: quaternion space dimension
        dropout: dropout rate
        cross_layer_rotation: add Cayley rotation between layers
        vocab_size: if set, adds embedding + output head
    """
    def __init__(self, name, d_model=512, n_heads=8, n_layers=4,
                 n_anchors=32, manifold_dim=256, n_comp=8, d_comp=32,
                 context_dim=128, quat_dim=64, dropout=0.1,
                 cross_layer_rotation=True, vocab_size=None, max_seq_len=2048):
        super().__init__(name)
        self.d_model = d_model
        self.n_layers = n_layers

        if vocab_size is not None:
            self.attach('embed', nn.Embedding(vocab_size, d_model))
            self.attach('pos_embed', nn.Embedding(max_seq_len, d_model))
            self.attach('head', nn.Linear(d_model, vocab_size, bias=False))

        for i in range(n_layers):
            self.attach(f'layer_{i}', GeometricTransformerLayer(
                f'{name}_L{i}', d_model, n_heads, n_anchors,
                manifold_dim, n_comp, d_comp, context_dim, quat_dim, dropout))

        if cross_layer_rotation and n_layers > 1:
            for i in range(n_layers - 1):
                self.attach(f'cross_rot_{i}', CayleyOrthogonal(
                    f'{name}_xrot_{i}', d_model))

        self.attach('final_norm', nn.LayerNorm(d_model))

        self._config = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            n_anchors=n_anchors, manifold_dim=manifold_dim,
            n_comp=n_comp, d_comp=d_comp, context_dim=context_dim,
            quat_dim=quat_dim, dropout=dropout,
            cross_layer_rotation=cross_layer_rotation,
            vocab_size=vocab_size,
        )

    @property
    def config(self):
        return self._config.copy()

    def param_report(self):
        total = 0
        name = getattr(self, '_tower_name', getattr(self, 'name', self.__class__.__name__))
        print(f"\n  {name} — parameter report")
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
        Args:
            x: (B, L, D) hidden states or (B, L) token ids
            return_geo_state: if True, return per-layer geometric state dicts

        Returns:
            out: (B, L, D) transformed hidden states (or logits if head attached)
            geo_states: list of per-layer geo_state dicts (if return_geo_state)
                Each dict contains: embedding, geo_ctx, triangulation,
                cos_to_anchors, assignment, nearest, patchwork, bridge,
                content, geometric, composed
        """
        if self.has('embed') and x.dtype in (torch.long, torch.int32, torch.int64):
            pos = torch.arange(x.shape[1], device=x.device)
            x = self['embed'](x) + self['pos_embed'](pos)

        geo_states = []
        has_xrot = self.has('cross_rot_0')

        for i in range(self.n_layers):
            x, geo_state = self[f'layer_{i}'](
                x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            if return_geo_state:
                geo_states.append(geo_state)
            if has_xrot and i < self.n_layers - 1:
                x = self[f'cross_rot_{i}'](x)

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
    print("Geometric Transformer — Self-Test")
    print(f"  geolip_core available: {_HAS_GEOLIP}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = geo_transformer_small('test', n_layers=2)
    if hasattr(model, 'network_to'):
        model.network_to(device=device, strict=False)
    else:
        model = model.to(device)
    total = model.param_report()

    B, L, D = 2, 32, 256
    x = torch.randn(B, L, D, device=device)

    out, geos = model(x, return_geo_state=True)
    assert out.shape == (B, L, D), f"Expected ({B},{L},{D}), got {out.shape}"
    assert len(geos) == 2

    print(f"\n  Input:  ({B}, {L}, {D})")
    print(f"  Output: {out.shape}")
    print(f"  Geo states: {len(geos)} layers")
    print(f"  State keys: {sorted(geos[0].keys())}")
    for k, v in geos[0].items():
        if v is not None:
            shape = v.shape if hasattr(v, 'shape') else type(v).__name__
            print(f"    {k:<18s}: {shape}")

    # Verify rotations
    for name, module in model.named_modules():
        if isinstance(module, CayleyOrthogonal):
            R = module.get_rotation()
            I = torch.eye(R.shape[0], device=R.device)
            print(f"  {name}: ‖RRᵀ-I‖={((R@R.T)-I).norm():.8f}  det={torch.det(R):.4f}")

    # ESM-2 scale overhead
    print(f"\n  ESM-2 scale:")
    esm = geo_transformer_esm2('esm2', n_layers=6)
    if hasattr(esm, 'network_to'):
        esm.network_to(device=device, strict=False)
    else:
        esm = esm.to(device)
    n = esm.param_report()
    print(f"  Overhead on 650M base: {n/1e6:.1f}M ({n/650e6*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"  PASSED")
    print(f"{'='*60}")