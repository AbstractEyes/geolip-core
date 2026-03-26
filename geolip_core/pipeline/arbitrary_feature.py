"""
Geometric Pipeline — geofractal router-based composable pipeline.

Uses TorchComponent wrappers from pipeline/components/. Each component
wraps a core module and communicates via the router's ephemeral cache.

Usage:
    from geolip_core.pipeline.geometric_pipeline import GeometricPipeline

    pipe = GeometricPipeline('geo', seq_len=5, input_dim=512)
    features = pipe(x)           # (B, 5, 512) → (B, feature_dim)
    pipe.get_diagnostics()       # all intermediates
    pipe.cache_clear()           # lifecycle managed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geofractal.router.base_tower import BaseTower

from geolip_core.pipeline.components import (
    ObserveSVDTokens, AssociateConstellation,
    CurateCMGate, CuratePatchwork, FuseGeometric,
)


class GeometricPipeline(BaseTower):
    """Composable geometric observation pipeline.

    A BaseTower that attaches geometric stage components and
    orchestrates their flow via cache.

    Stages communicate entirely through the router's ephemeral cache.
    Each stage reads what it needs, writes what it produces. The pipeline's
    forward() determines execution order. Swap any component to change
    behavior without touching the flow.

    Default configuration:
        observe     → ObserveSVDTokens      (structural decomposition)
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
                 activation='squared_relu', strict=False):
        super().__init__(name, strict=strict)

        pw_dim = n_comp * d_comp
        svd_feat_dim = 2 * seq_len + 2

        # Config as non-module object
        self.attach('config', {
            'seq_len': seq_len, 'input_dim': input_dim,
            'embed_dim': embed_dim, 'n_anchors': n_anchors,
            'n_comp': n_comp, 'd_comp': d_comp,
            'pw_dim': pw_dim, 'svd_feat_dim': svd_feat_dim,
            'gate_strategy': gate_strategy,
        })

        # ── Pipeline-specific projection: tokens → S^(embed_dim-1) ──
        self.attach('embed_proj', nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim)))

        # ── Stage components (all from pipeline/components/) ──
        self.attach('observe', ObserveSVDTokens(
            'observe', seq_len))

        self.attach('associate', AssociateConstellation(
            'associate', dim=embed_dim, n_anchors=n_anchors))

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
        self.cache_set('input', x)

        # ── INPUT: structural observation ──
        self['observe'](x)

        # ── ASSOCIATE: pool → project → sphere → triangulate ──
        pooled = x.mean(dim=1)
        emb = F.normalize(self['embed_proj'](pooled), dim=-1)
        self['associate'](emb)

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
    _check("has embed_proj", pipe.has('embed_proj'))
    _check("has config", pipe.has('config'))

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
    pipe.detach('curate_gate')
    pipe.attach('curate_gate', CurateCMGate(
        'curate_gate', 16, 128, 4, 2, 'round_robin').to(device))

    features_rr = pipe(x)
    rr_info = pipe.cache_get('gate_info')
    _check("swapped to round_robin", rr_info['strategy'] == 'round_robin')
    _check("still produces output", features_rr.shape == (B, pipe.feature_dim))

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
               for p in pipe['embed_proj'].parameters()))
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