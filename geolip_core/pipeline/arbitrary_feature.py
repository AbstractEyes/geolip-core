"""
Prototype pipeline: arbitrary (B, seq, dim) → unified geometric feature.

Takes any structured embedding sequence and processes it through:
  INPUT:     SVD observation of token structure (transpose → gram_eigh)
  ASSOCIATE: Pool → S^(d-1) → triangulate against constellation
  CURATE:    CM gate → gated patchwork interpretation
  OUTPUT:    Unified geometric feature vector

This is the minimal pipeline — no backbone, no task head, no loss.
Just: structured input → geometric feature.

Usage:
    from geolip_core_proto import GeometricFeaturizer

    featurizer = GeometricFeaturizer(seq_len=5, input_dim=512)
    features = featurizer(x)  # (B, 5, 512) → (B, feature_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import from the refactored package ──
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geolip_core.utils.kernel import gram_eigh_svd
from geolip_core.core.associate.constellation import Constellation, init_anchors_repulsion
from geolip_core.core.curate.gate import AnchorGate, cayley_menger_det
from geolip_core.core.util import make_activation


class SVDTokenObserver(nn.Module):
    """Observe structural relationships between tokens via SVD.

    Input: (B, seq, dim) — sequence of embeddings
    SVD on transposed (B, dim, seq) — 'dim' positions, 'seq' channels.
    Produces seq singular values + seq×seq rotation.

    For (B, 5, 512): transposes to (B, 512, 5), SVD gives S(5), Vh(5,5).
    The singular values tell you how much variance each token-direction captures.
    Vh tells you how the token dimensions mix in the feature space.
    """

    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        # Feature dim: S_norm(seq) + Vh_diag(seq) + offdiag(1) + entropy(1)
        self.feature_dim = 2 * seq_len + 2

    def forward(self, x):
        """(B, seq, dim) → S, Vh, features."""
        B, S, D = x.shape

        # Transpose: treat dim as spatial, seq as channels
        # (B, seq, dim) → (B, dim, seq) — now M=dim >> N=seq
        x_t = x.transpose(1, 2).contiguous()  # (B, dim, seq)

        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                _, sv, vh = gram_eigh_svd(x_t.float())
                sv = sv.clamp(min=1e-6)
                sv = torch.where(torch.isfinite(sv), sv, torch.ones_like(sv))
                vh = torch.where(torch.isfinite(vh), vh, torch.zeros_like(vh))

        # Compact features
        s_norm = sv / (sv.sum(dim=-1, keepdim=True) + 1e-8)
        vh_diag = vh.diagonal(dim1=-2, dim2=-1)
        vh_offdiag = (vh.pow(2).sum((-2, -1)) - vh_diag.pow(2).sum(-1)).unsqueeze(-1).clamp(min=0)
        s_ent = -(s_norm * torch.log(s_norm.clamp(min=1e-8))).sum(-1, keepdim=True)

        features = torch.cat([s_norm, vh_diag, vh_offdiag, s_ent], dim=-1)
        features = torch.where(torch.isfinite(features), features, torch.zeros_like(features))

        return sv, vh, features


class GeometricFeaturizer(nn.Module):
    """Complete pipeline: (B, seq, dim) → unified geometric feature.

    Stages:
      1. INPUT:     SVD of token structure → structural features
      2. ASSOCIATE: Mean-pool → S^(embed_dim-1) → triangulate
      3. CURATE:    CM gate → gated patchwork
      4. FUSE:      Concatenate SVD context + patchwork + embedding

    Args:
        seq_len:     Number of tokens in input sequence
        input_dim:   Dimension of each token embedding
        embed_dim:   Constellation embedding dimension
        n_anchors:   Number of constellation anchors
        n_comp:      Patchwork compartments
        d_comp:      Per-compartment hidden dim
        gate_strategy: 'round_robin', 'cm_gate', 'top_k', 'top_p'
        n_neighbors: CM simplex neighbors
    """

    def __init__(self, seq_len=5, input_dim=512, embed_dim=256,
                 n_anchors=32, n_comp=8, d_comp=32,
                 gate_strategy='cm_gate', n_neighbors=3):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_anchors = n_anchors
        pw_dim = n_comp * d_comp

        # ── INPUT: SVD observation ──
        self.svd_observer = SVDTokenObserver(seq_len)
        self.svd_proj = nn.Sequential(
            nn.Linear(self.svd_observer.feature_dim, pw_dim),
            nn.LayerNorm(pw_dim), nn.GELU())

        # ── ASSOCIATE: project + triangulate ──
        self.embed_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim))

        self.anchors = nn.Parameter(init_anchors_repulsion(n_anchors, embed_dim))

        # ── CURATE: CM gate + patchwork ──
        self.gate = AnchorGate(
            n_anchors, embed_dim, n_comp, n_neighbors, gate_strategy)

        self.comps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_anchors, d_comp * 2),
                nn.GELU(),
                nn.Linear(d_comp * 2, d_comp),
                nn.LayerNorm(d_comp))
            for _ in range(n_comp)])

        # ── OUTPUT: fuse all geometric signals ──
        # svd_context (pw_dim) + patchwork (pw_dim) + embedding (embed_dim)
        self._feature_dim = pw_dim + pw_dim + embed_dim

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward(self, x):
        """Process arbitrary token sequence into geometric feature.

        Args:
            x: (B, seq_len, input_dim) — any structured embedding sequence

        Returns:
            features: (B, feature_dim) — unified geometric feature
            info: dict with all intermediates and diagnostics
        """
        B, S, D = x.shape
        assert S == self.seq_len and D == self.input_dim, \
            f"Expected ({B}, {self.seq_len}, {self.input_dim}), got {x.shape}"

        # ════ INPUT: SVD structural observation ════
        sv, vh, svd_raw = self.svd_observer(x)
        svd_context = self.svd_proj(svd_raw)  # (B, pw_dim)

        # ════ ASSOCIATE: pool → sphere → triangulate ════
        # Mean-pool tokens, project, normalize to S^(embed_dim-1)
        pooled = x.mean(dim=1)  # (B, input_dim)
        emb = F.normalize(self.embed_proj(pooled), dim=-1)  # (B, embed_dim)

        anchors_n = F.normalize(self.anchors, dim=-1)
        cos = emb @ anchors_n.detach().T  # (B, A)
        tri = 1.0 - cos

        # ════ CURATE: CM gate + gated patchwork ════
        gate_values, gate_assign, gate_info = self.gate(emb, anchors_n.detach(), tri)
        tri_gated = tri * gate_values

        pw = torch.cat([comp(tri_gated) for comp in self.comps], dim=-1)  # (B, pw_dim)

        # ════ FUSE: concatenate all geometric signals ════
        features = torch.cat([svd_context, pw, emb], dim=-1)  # (B, feature_dim)

        info = {
            'embedding': emb,
            'singular_values': sv,
            'Vh': vh,
            'svd_features': svd_raw,
            'svd_context': svd_context,
            'cos': cos, 'tri': tri,
            'gate_values': gate_values,
            'gate_info': gate_info,
            'patchwork': pw,
        }

        return features, info


class GeometricClassifier(nn.Module):
    """Featurizer + task head. Drop-in for classification testing.

    Args:
        seq_len, input_dim: input shape
        num_classes: classification targets
        **kwargs: forwarded to GeometricFeaturizer
    """

    def __init__(self, seq_len=5, input_dim=512, num_classes=100, **kwargs):
        super().__init__()
        self.featurizer = GeometricFeaturizer(seq_len=seq_len, input_dim=input_dim, **kwargs)
        self.head = nn.Sequential(
            nn.Linear(self.featurizer.feature_dim, 256),
            nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.1),
            nn.Linear(256, num_classes))

    def forward(self, x):
        features, info = self.featurizer(x)
        logits = self.head(features)
        info['logits'] = logits
        return logits, info


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Geometric featurizer prototype — {device}")
    print()

    B, S, D = 16, 5, 512
    _counts = {'passed': 0, 'failed': 0}

    def _check(name, condition, detail=""):
        if condition:
            _counts['passed'] += 1; print(f"  [PASS] {name}")
        else:
            _counts['failed'] += 1; print(f"  [FAIL] {name}  {detail}")

    # ── 1. SVD token observer ──
    print("1. SVDTokenObserver:")
    obs = SVDTokenObserver(seq_len=S).to(device)
    x = torch.randn(B, S, D, device=device)
    sv, vh, feats = obs(x)
    _check("S shape", sv.shape == (B, S), f"got {sv.shape}")
    _check("Vh shape", vh.shape == (B, S, S), f"got {vh.shape}")
    _check("features shape", feats.shape == (B, 2*S+2), f"got {feats.shape}")
    _check("S positive", (sv > 0).all().item())
    _check("S descending", (sv[:, :-1] >= sv[:, 1:] - 1e-5).all().item())
    _check("features finite", torch.isfinite(feats).all().item())

    # ── 2. Full featurizer ──
    print("\n2. GeometricFeaturizer:")
    feat = GeometricFeaturizer(
        seq_len=S, input_dim=D, embed_dim=128,
        n_anchors=16, n_comp=4, d_comp=32,
        gate_strategy='cm_gate', n_neighbors=2).to(device)

    x = torch.randn(B, S, D, device=device)
    features, info = feat(x)

    _check("output shape", features.shape == (B, feat.feature_dim),
           f"got {features.shape}, expected (B, {feat.feature_dim})")
    _check("output finite", torch.isfinite(features).all().item())
    _check("embedding on sphere",
           (info['embedding'].norm(dim=-1) - 1.0).abs().max().item() < 1e-5)
    _check("gate values in [0,1]",
           info['gate_values'].min() >= 0 and info['gate_values'].max() <= 1.01)

    print(f"\n  Feature breakdown:")
    pw_dim = 4 * 32
    print(f"    SVD context:  {pw_dim}")
    print(f"    Patchwork:    {pw_dim}")
    print(f"    Embedding:    128")
    print(f"    Total:        {feat.feature_dim}")
    print(f"  Gate info: {info['gate_info']}")
    print(f"  Singular values (sample 0): {info['singular_values'][0].tolist()}")

    # ── 3. Different gate strategies ──
    print("\n3. Gate strategies:")
    for strategy in ['round_robin', 'cm_gate', 'top_k', 'top_p']:
        f = GeometricFeaturizer(
            seq_len=S, input_dim=D, embed_dim=128,
            n_anchors=16, n_comp=4, d_comp=32,
            gate_strategy=strategy, n_neighbors=2).to(device)
        out, inf = f(x)
        _check(f"{strategy}: shape + finite",
               out.shape == (B, f.feature_dim) and torch.isfinite(out).all().item())

    # ── 4. Gradient flow ──
    print("\n4. Gradient flow:")
    feat_grad = GeometricFeaturizer(
        seq_len=S, input_dim=D, embed_dim=128,
        n_anchors=16, n_comp=4, d_comp=32,
        gate_strategy='cm_gate', n_neighbors=2).to(device)

    x_grad = torch.randn(B, S, D, device=device, requires_grad=True)
    out, _ = feat_grad(x_grad)
    loss = out.sum()
    loss.backward()
    _check("gradient reaches input", x_grad.grad is not None and x_grad.grad.abs().sum() > 0)
    _check("gradient reaches embed_proj",
           any(p.grad is not None and p.grad.abs().sum() > 0
               for p in feat_grad.embed_proj.parameters()))
    _check("gradient reaches patchwork",
           any(p.grad is not None and p.grad.abs().sum() > 0
               for p in feat_grad.comps.parameters()))

    # ── 5. Classifier wrapper ──
    print("\n5. GeometricClassifier:")
    clf = GeometricClassifier(
        seq_len=S, input_dim=D, num_classes=100,
        embed_dim=128, n_anchors=16, n_comp=4, d_comp=32,
        gate_strategy='cm_gate', n_neighbors=2).to(device)

    x = torch.randn(B, S, D, device=device)
    labels = torch.randint(0, 100, (B,), device=device)
    logits, info = clf(x)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    _check("logits shape", logits.shape == (B, 100))
    _check("loss finite", torch.isfinite(loss).item())
    _check("loss scalar", loss.dim() == 0)
    n_params = sum(p.numel() for p in clf.parameters())
    print(f"\n  Classifier params: {n_params:,}")
    print(f"  Loss: {loss.item():.4f}")

    # ── 6. Different input types (simulate real use cases) ──
    print("\n6. Simulated inputs:")

    # 5 expert embeddings (different models looking at same input)
    experts = torch.randn(B, 5, 512, device=device)
    out_exp, _ = feat(experts)
    _check("expert embeddings", torch.isfinite(out_exp).all().item())

    # 5 text tokens
    tokens = F.normalize(torch.randn(B, 5, 512, device=device), dim=-1)
    out_tok, _ = feat(tokens)
    _check("text tokens (unit norm)", torch.isfinite(out_tok).all().item())

    # 5 identical vectors (degenerate — SVD should still work)
    identical = torch.randn(B, 1, 512, device=device).expand(B, 5, 512).contiguous()
    out_ident, info_ident = feat(identical)
    _check("identical tokens (degenerate)", torch.isfinite(out_ident).all().item())
    # S should have one dominant value, rest near zero
    s_ratio = info_ident['singular_values'][:, 0] / (info_ident['singular_values'].sum(-1) + 1e-8)
    _check("degenerate: S[0] dominates", s_ratio.mean().item() > 0.9,
           f"ratio={s_ratio.mean().item():.3f}")

    # 5 orthogonal vectors (maximally spread)
    ortho_base = torch.linalg.qr(torch.randn(512, 5, device=device)).Q.T  # (5, 512)
    ortho = ortho_base.unsqueeze(0).expand(B, -1, -1).contiguous()
    out_orth, info_orth = feat(ortho)
    _check("orthogonal tokens", torch.isfinite(out_orth).all().item())
    s_std = info_orth['singular_values'].std(dim=-1).mean().item()
    _check("orthogonal: S roughly equal", s_std < 0.5,
           f"std={s_std:.3f}")

    # ── Summary ──
    total = _counts['passed'] + _counts['failed']
    print(f"\n{'='*50}")
    print(f"  {_counts['passed']}/{total} passed" +
          (f"  ({_counts['failed']} FAILED)" if _counts['failed'] else " — all clear"))
    print(f"{'='*50}")