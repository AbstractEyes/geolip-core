"""
Geometric Pipeline — geofractal router-based composable pipeline.

Each geometric behavior is a TorchComponent. The pipeline is a BaseTower
that composes them. Stages communicate via the router's ephemeral cache.

Attach, swap, reorder. The pipeline is the composition. The model
is what you build FROM the pipeline.

Usage:
    from geolip_core.pipeline.geometric_pipeline import (
        GeometricPipeline, ObserveSVD, AssociateConstellation,
        CurateCMGate, CuratePatchwork, FuseGeometric,
    )

    pipe = GeometricPipeline('geo', seq_len=5, input_dim=512)
    features = pipe(x)           # (B, 5, 512) → (B, feature_dim)
    pipe.cache_get('gate_info')  # access intermediates

    # Swap curation strategy
    pipe.detach('curate_gate')
    pipe.attach('curate_gate', CurateCMGate('curate_gate', ...))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geofractal.router.base_tower import BaseTower
from geofractal.router.components.torch_component import TorchComponent

from geolip_core.utils.kernel import gram_eigh_svd
from geolip_core.core.associate.constellation import init_anchors_repulsion
from geolip_core.core.curate.gate import AnchorGate
from geolip_core.core.util import make_activation


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE COMPONENTS — each a TorchComponent, communicates via parent cache
# ═══════════════════════════════════════════════════════════════════════════════

class ObserveSVD(TorchComponent):
    """INPUT stage: SVD structural observation.

    Reads:  cache['input']        — (B, seq, dim) raw input
    Writes: cache['svd_S']        — (B, seq) singular values
            cache['svd_Vh']       — (B, seq, seq) rotation matrix
            cache['svd_features'] — (B, 2*seq+2) compact summary
    """

    def __init__(self, name, seq_len, **kwargs):
        super().__init__(name, **kwargs)
        self.seq_len = seq_len
        self.feature_dim = 2 * seq_len + 2

    def forward(self, x):
        """(B, seq, dim) → writes SVD decomposition to parent cache."""
        B, S, D = x.shape
        x_t = x.transpose(1, 2).contiguous()

        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                _, sv, vh = gram_eigh_svd(x_t.float())
                sv = sv.clamp(min=1e-6)
                sv = torch.where(torch.isfinite(sv), sv, torch.ones_like(sv))
                vh = torch.where(torch.isfinite(vh), vh, torch.zeros_like(vh))

        s_norm = sv / (sv.sum(dim=-1, keepdim=True) + 1e-8)
        vh_diag = vh.diagonal(dim1=-2, dim2=-1)
        vh_offdiag = (vh.pow(2).sum((-2, -1)) - vh_diag.pow(2).sum(-1)).unsqueeze(-1).clamp(min=0)
        s_ent = -(s_norm * torch.log(s_norm.clamp(min=1e-8))).sum(-1, keepdim=True)
        features = torch.cat([s_norm, vh_diag, vh_offdiag, s_ent], dim=-1)
        features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))

        if self.parent is not None:
            self.parent.cache_set('svd_S', sv)
            self.parent.cache_set('svd_Vh', vh)
            self.parent.cache_set('svd_features', features)

        return features


class AssociateConstellation(TorchComponent):
    """ASSOCIATE stage: project to sphere, triangulate against anchors.

    Reads:  cache['input'] — (B, seq, dim) raw input
    Writes: cache['embedding']   — (B, embed_dim) on S^(d-1)
            cache['anchors_n']   — (A, embed_dim) normalized anchors
            cache['cos']         — (B, A) cosine similarities
            cache['tri']         — (B, A) triangulation distances
    """

    def __init__(self, name, input_dim, embed_dim=256, n_anchors=32, **kwargs):
        super().__init__(name, **kwargs)
        self.embed_dim = embed_dim
        self.n_anchors = n_anchors

        self.embed_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim))

        self.anchors = nn.Parameter(init_anchors_repulsion(n_anchors, embed_dim))

    def forward(self, x):
        """(B, seq, dim) → writes embedding + triangulation to parent cache."""
        pooled = x.mean(dim=1)
        emb = F.normalize(self.embed_proj(pooled), dim=-1)
        anchors_n = F.normalize(self.anchors, dim=-1)

        cos = emb @ anchors_n.detach().T
        tri = 1.0 - cos

        if self.parent is not None:
            self.parent.cache_set('embedding', emb)
            self.parent.cache_set('anchors_n', anchors_n)
            self.parent.cache_set('cos', cos)
            self.parent.cache_set('tri', tri)

        return tri


class CurateCMGate(TorchComponent):
    """CURATE stage (gate): CM validity selection.

    Reads:  cache['embedding']  — (B, embed_dim)
            cache['anchors_n']  — (A, embed_dim)
            cache['tri']        — (B, A)
    Writes: cache['gate_values'] — (B, A) per-anchor gate
            cache['gate_info']   — dict diagnostics
            cache['tri_gated']   — (B, A) gated triangulation
    """

    def __init__(self, name, n_anchors, embed_dim, n_comp=8,
                 n_neighbors=3, strategy='cm_gate', **kwargs):
        super().__init__(name, **kwargs)
        self.gate = AnchorGate(
            n_anchors, embed_dim, n_comp, n_neighbors, strategy)

    def forward(self, tri=None):
        """Read from cache, gate, write back."""
        emb = self.parent.cache_get('embedding')
        anchors_n = self.parent.cache_get('anchors_n')
        if tri is None:
            tri = self.parent.cache_get('tri')

        gate_values, gate_assign, gate_info = self.gate(emb, anchors_n.detach(), tri)
        tri_gated = tri * gate_values

        self.parent.cache_set('gate_values', gate_values)
        self.parent.cache_set('gate_info', gate_info)
        self.parent.cache_set('tri_gated', tri_gated)

        return tri_gated


class CuratePatchwork(TorchComponent):
    """CURATE stage (interpret): compartmentalized patchwork.

    Reads:  cache['tri_gated'] — (B, A) gated triangulation
    Writes: cache['patchwork']  — (B, pw_dim) interpreted features
    """

    def __init__(self, name, n_anchors, n_comp=8, d_comp=32,
                 activation='gelu', **kwargs):
        super().__init__(name, **kwargs)
        self.pw_dim = n_comp * d_comp

        self.comps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_anchors, d_comp * 2),
                make_activation(activation),
                nn.Linear(d_comp * 2, d_comp),
                nn.LayerNorm(d_comp))
            for _ in range(n_comp)])

    def forward(self, tri_gated=None):
        """Read gated triangulation, interpret through compartments."""
        if tri_gated is None:
            tri_gated = self.parent.cache_get('tri_gated')

        pw = torch.cat([comp(tri_gated) for comp in self.comps], dim=-1)

        if self.parent is not None:
            self.parent.cache_set('patchwork', pw)

        return pw


class FuseGeometric(TorchComponent):
    """OUTPUT stage: fuse all geometric signals into unified feature.

    Reads:  cache['svd_features'] — (B, svd_feat_dim)
            cache['patchwork']    — (B, pw_dim)
            cache['embedding']    — (B, embed_dim)
    Writes: cache['geo_features'] — (B, feature_dim) unified output
    """

    def __init__(self, name, svd_feat_dim, pw_dim, embed_dim, **kwargs):
        super().__init__(name, **kwargs)
        self.svd_proj = nn.Sequential(
            nn.Linear(svd_feat_dim, pw_dim),
            nn.LayerNorm(pw_dim), nn.GELU())

        self.feature_dim = pw_dim + pw_dim + embed_dim

    def forward(self):
        """Fuse all cached geometric signals."""
        svd_raw = self.parent.cache_get('svd_features')
        pw = self.parent.cache_get('patchwork')
        emb = self.parent.cache_get('embedding')

        svd_context = self.svd_proj(svd_raw)
        features = torch.cat([svd_context, pw, emb], dim=-1)

        self.parent.cache_set('svd_context', svd_context)
        self.parent.cache_set('geo_features', features)

        return features


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC PIPELINE — BaseTower composing stage components
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricPipeline(BaseTower):
    """Composable geometric observation pipeline.

    A BaseTower that attaches geometric stage components and
    orchestrates their flow via cache.

    Stages communicate entirely through the router's ephemeral cache.
    Each stage reads what it needs, writes what it produces. The pipeline's
    forward() determines execution order. Swap any component to change
    behavior without touching the flow.

    Default configuration:
        observe     → ObserveSVD            (structural decomposition)
        associate   → AssociateConstellation (anchor triangulation)
        curate_gate → CurateCMGate          (CM validity selection)
        curate_pw   → CuratePatchwork       (compartment interpretation)
        fuse        → FuseGeometric         (unified feature output)

    Args:
        name:       Router name
        seq_len:    Input sequence length
        input_dim:  Input embedding dimension
        embed_dim:  Constellation embedding dimension
        n_anchors:  Number of constellation anchors
        n_comp:     Patchwork compartments
        d_comp:     Per-compartment hidden dim
        gate_strategy: CM gate strategy
        n_neighbors:   CM simplex neighbors
    """

    def __init__(self, name, seq_len=5, input_dim=512, embed_dim=256,
                 n_anchors=32, n_comp=8, d_comp=32,
                 gate_strategy='cm_gate', n_neighbors=3,
                 activation='gelu', strict=False):
        super().__init__(name, strict=strict)

        pw_dim = n_comp * d_comp
        svd_feat_dim = 2 * seq_len + 2

        # Store config as non-module object
        self.attach('config', {
            'seq_len': seq_len, 'input_dim': input_dim,
            'embed_dim': embed_dim, 'n_anchors': n_anchors,
            'n_comp': n_comp, 'd_comp': d_comp,
            'pw_dim': pw_dim, 'svd_feat_dim': svd_feat_dim,
            'gate_strategy': gate_strategy,
        })

        # ── Attach stage components ──
        self.attach('observe', ObserveSVD(
            'observe', seq_len))

        self.attach('associate', AssociateConstellation(
            'associate', input_dim, embed_dim, n_anchors))

        self.attach('curate_gate', CurateCMGate(
            'curate_gate', n_anchors, embed_dim, n_comp,
            n_neighbors, gate_strategy))

        self.attach('curate_pw', CuratePatchwork(
            'curate_pw', n_anchors, n_comp, d_comp, activation))

        self.attach('fuse', FuseGeometric(
            'fuse', svd_feat_dim, pw_dim, embed_dim))

    @property
    def feature_dim(self):
        return self['fuse'].feature_dim

    def forward(self, x):
        """Execute geometric pipeline.

        Args:
            x: (B, seq_len, input_dim) — structured embedding sequence

        Returns:
            features: (B, feature_dim) — unified geometric feature
        """
        # Store input in cache for stages that need it
        self.cache_set('input', x)

        # ── INPUT: structural observation (parallel-safe) ──
        self['observe'](x)

        # ── ASSOCIATE: sphere projection + triangulation ──
        self['associate'](x)

        # ── CURATE: gate → patchwork ──
        self['curate_gate']()
        self['curate_pw']()

        # ── FUSE: combine all geometric signals ──
        features = self['fuse']()

        return features

    def get_diagnostics(self):
        """Retrieve all diagnostic info from last forward pass."""
        return {
            'svd_S': self.cache_get('svd_S'),
            'svd_Vh': self.cache_get('svd_Vh'),
            'embedding': self.cache_get('embedding'),
            'cos': self.cache_get('cos'),
            'tri': self.cache_get('tri'),
            'gate_values': self.cache_get('gate_values'),
            'gate_info': self.cache_get('gate_info'),
            'patchwork': self.cache_get('patchwork'),
            'svd_context': self.cache_get('svd_context'),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GeometricPipeline (geofractal router) — {device}")
    print()

    B, S, D = 16, 5, 512
    _counts = {'passed': 0, 'failed': 0}

    def _check(name, condition, detail=""):
        if condition:
            _counts['passed'] += 1; print(f"  [PASS] {name}")
        else:
            _counts['failed'] += 1; print(f"  [FAIL] {name}  {detail}")

    # ── 1. Pipeline construction ──
    print("1. Pipeline construction:")
    pipe = GeometricPipeline(
        'geo', seq_len=S, input_dim=D, embed_dim=128,
        n_anchors=16, n_comp=4, d_comp=32,
        gate_strategy='cm_gate', n_neighbors=2).to(device)

    _check("has observe", pipe.has('observe'))
    _check("has associate", pipe.has('associate'))
    _check("has curate_gate", pipe.has('curate_gate'))
    _check("has curate_pw", pipe.has('curate_pw'))
    _check("has fuse", pipe.has('fuse'))
    _check("has config", pipe.has('config'))
    _check("config is object", isinstance(pipe['config'], dict))

    n_params = sum(p.numel() for p in pipe.parameters())
    print(f"  Params: {n_params:,}")

    # ── 2. Forward pass ──
    print("\n2. Forward pass:")
    x = torch.randn(B, S, D, device=device)
    features = pipe(x)

    _check("output shape", features.shape == (B, pipe.feature_dim),
           f"got {features.shape}")
    _check("output finite", torch.isfinite(features).all().item())

    # ── 3. Cache populated ──
    print("\n3. Cache after forward:")
    diag = pipe.get_diagnostics()
    _check("svd_S cached", diag['svd_S'] is not None and diag['svd_S'].shape == (B, S))
    _check("embedding cached", diag['embedding'] is not None)
    _check("embedding on sphere",
           (diag['embedding'].norm(dim=-1) - 1.0).abs().max().item() < 1e-5)
    _check("gate_values cached", diag['gate_values'] is not None)
    _check("patchwork cached", diag['patchwork'] is not None)
    _check("gate_info has strategy", diag['gate_info'].get('strategy') == 'cm_gate')

    # ── 4. Cache clear ──
    print("\n4. Cache lifecycle:")
    pipe.cache_clear()
    _check("cache cleared", pipe.cache_get('svd_S') is None)

    # ── 5. Component swap ──
    print("\n5. Component swap:")
    # Swap gate strategy: cm_gate → round_robin
    pipe.detach('curate_gate')
    pipe.attach('curate_gate', CurateCMGate(
        'curate_gate', 16, 128, 4, 2, 'round_robin').to(device))

    features_rr = pipe(x)
    rr_info = pipe.cache_get('gate_info')
    _check("swapped to round_robin", rr_info['strategy'] == 'round_robin')
    _check("still produces output", features_rr.shape == (B, pipe.feature_dim))

    # Swap back
    pipe.detach('curate_gate')
    pipe.attach('curate_gate', CurateCMGate(
        'curate_gate', 16, 128, 4, 2, 'top_k').to(device))

    features_topk = pipe(x)
    topk_info = pipe.cache_get('gate_info')
    _check("swapped to top_k", topk_info['strategy'] == 'top_k')

    # ── 6. Gradient flow ──
    print("\n6. Gradient flow:")
    pipe.cache_clear()
    x_grad = torch.randn(B, S, D, device=device, requires_grad=True)
    out = pipe(x_grad)
    loss = out.sum()
    loss.backward()

    _check("grad reaches input", x_grad.grad is not None and x_grad.grad.abs().sum() > 0)
    _check("grad reaches embed_proj",
           any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pipe['associate'].parameters()))
    _check("grad reaches patchwork",
           any(p.grad is not None and p.grad.abs().sum() > 0
               for p in pipe['curate_pw'].parameters()))

    # ── 7. Router introspection ──
    print("\n7. Router introspection:")
    print(f"  {repr(pipe)[:500]}...")
    cache_bytes = pipe.cache_size_bytes()
    print(f"  Cache size: {cache_bytes / 1024:.1f} KB")
    print(f"  Cache keys: {pipe.cache_keys()}")

    # ── Summary ──
    total = _counts['passed'] + _counts['failed']
    print(f"\n{'='*50}")
    print(f"  {_counts['passed']}/{total} passed" +
          (f"  ({_counts['failed']} FAILED)" if _counts['failed'] else " — all clear"))
    print(f"{'='*50}")