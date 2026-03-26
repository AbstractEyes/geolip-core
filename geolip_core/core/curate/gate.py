"""
anchor_gate.py — Cayley-Menger validity gate for anchor selection.

Replaces round-robin anchor assignment in the patchwork with a learned gate
that detects whether each anchor forms a geometrically valid simplex with
the input embedding.

CM validity: given k+1 points, the Cayley-Menger determinant is positive
iff the points form a non-degenerate simplex. Negative or zero = collapsed
(collinear, coplanar, duplicate). The gate uses this as a structural prior:
only anchors that form valid simplices with the embedding contribute.

Selection strategies:
  round_robin:   Fixed assignment (baseline, current patchwork)
  cm_gate:       Soft gate based on CM validity score
  top_k:         Hard selection of k highest-validity anchors
  top_p:         Nucleus selection — cumulative validity until threshold p
  multi_select:  Each compartment selects its own top-k from all anchors

Author: AbstractPhil + Claude Opus 4.6
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# CAYLEY-MENGER VALIDITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def pairwise_distances_squared(points):
    """Batched pairwise squared distances. (B, N, D) → (B, N, N)."""
    gram = torch.bmm(points, points.transpose(1, 2))
    diag = gram.diagonal(dim1=-2, dim2=-1)
    return diag.unsqueeze(2) + diag.unsqueeze(1) - 2 * gram


def cayley_menger_det(points):
    """Cayley-Menger signed volume² for simplices. (B, K, D) → (B,).

    K = number of vertices (k+1 for a k-simplex).
    Sign-corrected so positive always means valid:
      Positive → non-degenerate simplex
      Zero → degenerate (coplanar/collinear)

    The raw det(M) alternates sign with simplex dimension k.
    Correction factor (-1)^(k+1) normalizes so valid = positive.
    """
    B, K, D = points.shape
    d2 = pairwise_distances_squared(points)  # (B, K, K)

    # Build CM matrix: (K+1) × (K+1)
    M = torch.zeros(B, K + 1, K + 1, device=points.device, dtype=points.dtype)
    M[:, 0, 1:] = 1.0
    M[:, 1:, 0] = 1.0
    M[:, 1:, 1:] = d2

    # Signed volume²: (-1)^(k+1) / (2^k * (k!)²) * det(M), where k = K-1
    # Raw det alternates sign: K=3 (triangle) → det<0 for valid,
    # K=4 (tet) → det>0 for valid. Apply (-1)^(k+1) so positive = valid always.
    raw = torch.linalg.det(M)
    k = K - 1
    sign = (-1.0) ** (k + 1)
    return sign * raw


def cm_validity_score(embedding, anchors, n_neighbors=3):
    """Per-anchor CM validity score.

    For each anchor, forms a simplex from:
      [embedding, anchor, n_neighbors nearest other anchors]
    Computes CM determinant. Positive = valid simplex.

    Args:
        embedding: (B, D) — input on S^(d-1)
        anchors:   (A, D) — anchor positions on S^(d-1)
        n_neighbors: how many extra anchors to include in each simplex

    Returns:
        validity: (B, A) — CM validity score per anchor
        raw_det:  (B, A) — raw CM determinant (for diagnostics)
    """
    B, D = embedding.shape
    A = anchors.shape[0]

    # Pairwise distances between anchors: (A, A)
    anchor_dists = torch.cdist(anchors.unsqueeze(0), anchors.unsqueeze(0)).squeeze(0)
    # Mask self
    anchor_dists.fill_diagonal_(float('inf'))

    # For each anchor, find n_neighbors nearest anchors
    _, nn_idx = anchor_dists.topk(n_neighbors, largest=False)  # (A, n_neighbors)

    # Embedding-to-anchor distances: (B, A)
    emb_anchor_dist = torch.cdist(embedding.unsqueeze(1), anchors.unsqueeze(0).expand(B, -1, -1))
    emb_anchor_dist = emb_anchor_dist.squeeze(1)  # (B, A)

    # Build simplices: for each (batch, anchor), form [emb, anchor, neighbors]
    # Total simplex size: 1 + 1 + n_neighbors = n_neighbors + 2
    K = n_neighbors + 2

    raw_dets = torch.zeros(B, A, device=embedding.device)

    # Vectorized over batch, loop over anchors (A is small, typically 16-512)
    for a in range(A):
        # Simplex vertices: [embedding, anchor_a, neighbor_1, ..., neighbor_n]
        neighbor_ids = nn_idx[a]  # (n_neighbors,)
        # Gather points: (B, K, D)
        pts = torch.stack([
            embedding,                              # (B, D)
            anchors[a].unsqueeze(0).expand(B, -1),  # (B, D)
            *[anchors[nid].unsqueeze(0).expand(B, -1) for nid in neighbor_ids]
        ], dim=1)  # (B, K, D)

        raw_dets[:, a] = cayley_menger_det(pts)

    # Normalize: the sign matters (positive = valid), magnitude varies with scale
    # Use sign * log(|det| + eps) for stable gating signal
    sign = raw_dets.sign()
    log_mag = torch.log(raw_dets.abs() + 1e-12)
    validity = sign * log_mag

    return validity, raw_dets


# ═══════════════════════════════════════════════════════════════════════════════
# ANCHOR GATE
# ═══════════════════════════════════════════════════════════════════════════════

class AnchorGate(nn.Module):
    """Learned gate that selects anchors based on CM validity.

    Combines raw CM validity score with a learned projection
    to produce per-anchor gate values.

    Strategies:
      'round_robin': Fixed assignment (ignores validity, baseline)
      'cm_gate':     Soft gate — sigmoid(learned(validity_features))
      'top_k':       Hard selection of top-k highest gates per sample
      'top_p':       Nucleus — cumulative gate mass until threshold p
      'multi_select': Per-compartment independent top-k
    """

    def __init__(self, n_anchors, dim, n_comp=8, n_neighbors=3,
                 strategy='cm_gate', top_k=None, top_p=0.9):
        super().__init__()
        self.n_anchors = n_anchors
        self.dim = dim
        self.n_comp = n_comp
        self.n_neighbors = n_neighbors
        self.strategy = strategy
        self.top_k = top_k or n_anchors
        self.top_p = top_p

        # Learned gate: validity_features → gate_value
        # Input per anchor: validity_score(1) + anchor_cos(1) + anchor_dist_rank(1) = 3
        gate_input_dim = 3
        self.gate_proj = nn.Sequential(
            nn.Linear(gate_input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # Initialize gate bias positive → sigmoid(+2) ≈ 0.88
        # Gates start OPEN. Training closes the ones that should be closed.
        # This is architecture-before-loss: don't learn to open, learn to close.
        nn.init.zeros_(self.gate_proj[2].weight)
        nn.init.constant_(self.gate_proj[2].bias, 2.0)

        # Compartment assignment for multi_select
        if strategy == 'multi_select':
            self.comp_queries = nn.Parameter(torch.randn(n_comp, dim) * 0.02)

        # Round-robin assignment (baseline)
        self.register_buffer('rr_assign', torch.arange(n_anchors) % n_comp)

    def _compute_gate_features(self, embedding, anchors):
        """Build per-anchor gate features.

        Returns: (B, A, 3) — [validity_score, cosine_sim, distance_rank]
        """
        B, D = embedding.shape
        A = anchors.shape[0]

        # CM validity
        validity, raw_det = cm_validity_score(embedding, anchors, self.n_neighbors)
        # Normalize validity to roughly [-1, 1]
        v_std = validity.std(dim=-1, keepdim=True).clamp(min=1e-8)
        validity_norm = (validity - validity.mean(dim=-1, keepdim=True)) / v_std

        # Cosine similarity to each anchor
        cos_sim = embedding @ anchors.T  # (B, A)

        # Distance rank (0=nearest, 1=farthest, normalized)
        dists = 1.0 - cos_sim  # angular distance proxy
        ranks = dists.argsort(dim=-1).argsort(dim=-1).float()  # (B, A)
        ranks = ranks / (A - 1)  # normalize to [0, 1]

        return torch.stack([validity_norm, cos_sim, ranks], dim=-1), raw_det

    def forward(self, embedding, anchors, tri):
        """Compute anchor gate values.

        Args:
            embedding: (B, D) — normalized embedding
            anchors:   (A, D) — normalized anchor positions
            tri:       (B, A) — triangulation (1 - cos_sim), already computed

        Returns:
            gate_values: (B, A) — per-anchor gate in [0, 1]
            assignment:  (B, A) → compartment_id or None (for round_robin)
            info:        dict with diagnostics
        """
        B = embedding.shape[0]
        A = anchors.shape[0]

        if self.strategy == 'round_robin':
            # Fixed assignment, all gates = 1
            gate_values = torch.ones(B, A, device=embedding.device)
            return gate_values, self.rr_assign.unsqueeze(0).expand(B, -1), {
                'strategy': 'round_robin', 'active': A}

        # Compute gate features
        gate_features, raw_det = self._compute_gate_features(embedding, anchors)
        # (B, A, 3) → (B, A, 1) → (B, A)
        raw_gate = self.gate_proj(gate_features).squeeze(-1)

        if self.strategy == 'cm_gate':
            gate_values = torch.sigmoid(raw_gate)
            active = (gate_values > 0.5).float().sum(-1).mean().item()
            return gate_values, None, {
                'strategy': 'cm_gate', 'active': active,
                'gate_mean': gate_values.mean().item(),
                'cm_positive_frac': (raw_det > 0).float().mean().item()}

        elif self.strategy == 'top_k':
            k = min(self.top_k, A)
            scores = raw_gate  # (B, A)
            topk_vals, topk_idx = scores.topk(k, dim=-1)
            # Sparse gate: 1 for selected, 0 for rest
            gate_values = torch.zeros(B, A, device=embedding.device)
            gate_values.scatter_(1, topk_idx, torch.sigmoid(topk_vals))
            return gate_values, topk_idx, {
                'strategy': 'top_k', 'k': k,
                'cm_positive_frac': (raw_det > 0).float().mean().item()}

        elif self.strategy == 'top_p':
            scores = torch.sigmoid(raw_gate)  # (B, A)
            sorted_scores, sorted_idx = scores.sort(dim=-1, descending=True)
            cumsum = sorted_scores.cumsum(dim=-1)
            # Mask: keep until cumulative mass exceeds p * total
            total = scores.sum(dim=-1, keepdim=True)
            mask = cumsum <= self.top_p * total
            # Always keep at least one
            mask[:, 0] = True
            gate_values = torch.zeros(B, A, device=embedding.device)
            gate_values.scatter_(1, sorted_idx, sorted_scores * mask.float())
            active = mask.float().sum(-1).mean().item()
            return gate_values, sorted_idx, {
                'strategy': 'top_p', 'p': self.top_p, 'active': active,
                'cm_positive_frac': (raw_det > 0).float().mean().item()}

        elif self.strategy == 'multi_select':
            # Each compartment independently selects anchors via attention
            comp_q = F.normalize(self.comp_queries, dim=-1)  # (n_comp, D)
            # Compartment-anchor affinity
            comp_anchor_affinity = comp_q @ anchors.T  # (n_comp, A)
            # Modulate by per-sample validity
            scores = torch.sigmoid(raw_gate).unsqueeze(1) * comp_anchor_affinity.unsqueeze(0)
            # (B, n_comp, A) — per-compartment per-anchor scores
            apc = max(A // self.n_comp, 2)
            topk_vals, topk_idx = scores.topk(apc, dim=-1)  # (B, n_comp, apc)
            return scores.mean(dim=1), topk_idx, {
                'strategy': 'multi_select', 'apc': apc,
                'cm_positive_frac': (raw_det > 0).float().mean().item()}

        raise ValueError(f"Unknown strategy: {self.strategy}")


# ═══════════════════════════════════════════════════════════════════════════════
# GATED PATCHWORK (replaces round-robin)
# ═══════════════════════════════════════════════════════════════════════════════

class GatedPatchwork(nn.Module):
    """Patchwork with CM validity anchor gating.

    Instead of round-robin assigning anchors to compartments, each anchor's
    contribution is weighted by its CM validity gate. Invalid anchors
    (degenerate simplex with the embedding) are suppressed.

    Args:
        n_anchors:  Number of anchors
        n_comp:     Number of compartments
        d_comp:     Output dim per compartment
        dim:        Embedding dimension (for gate computation)
        strategy:   Gate strategy ('round_robin', 'cm_gate', 'top_k', 'top_p')
        n_neighbors: Simplex neighbors for CM computation
        activation: Activation function name
    """

    def __init__(self, n_anchors, n_comp=8, d_comp=64, dim=256,
                 strategy='cm_gate', n_neighbors=3, activation='squared_relu',
                 top_k=None, top_p=0.9):
        super().__init__()
        self.n_comp = n_comp
        self.d_comp = d_comp
        self.n_anchors = n_anchors
        self.output_dim = n_comp * d_comp

        # Gate
        self.gate = AnchorGate(
            n_anchors, dim, n_comp, n_neighbors, strategy, top_k, top_p)

        # Compartment MLPs — each reads ALL anchors (gated)
        _activations = {
            'squared_relu': lambda: nn.Sequential(nn.ReLU(), nn.ReLU()),
            'gelu': lambda: nn.GELU(),
            'relu': lambda: nn.ReLU(),
        }
        make_act = _activations.get(activation, lambda: nn.GELU())

        self.comps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_anchors, d_comp * 2),
                make_act(),
                nn.Linear(d_comp * 2, d_comp),
                nn.LayerNorm(d_comp))
            for _ in range(n_comp)])

    def forward(self, tri, embedding=None, anchors=None):
        """Forward with optional gating.

        If embedding and anchors are provided, computes CM gate.
        Otherwise falls back to ungated (all weights = 1).

        Args:
            tri:       (B, A) — triangulation distances
            embedding: (B, D) — optional, for gate computation
            anchors:   (A, D) — optional, for gate computation

        Returns:
            output: (B, n_comp * d_comp)
            gate_info: dict with diagnostics
        """
        B, A = tri.shape

        if embedding is not None and anchors is not None:
            gate_values, assignment, gate_info = self.gate(embedding, anchors, tri)
            # Apply gate: weight triangulation distances by validity
            tri_gated = tri * gate_values  # (B, A) — invalid anchors suppressed
        else:
            tri_gated = tri
            gate_info = {'strategy': 'ungated'}

        # Each compartment reads the full gated triangulation
        out = torch.cat([comp(tri_gated) for comp in self.comps], dim=-1)
        return out, gate_info


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"anchor_gate.py — validation on {device}")
    print()

    B, D, A = 32, 64, 16
    _counts = {'passed': 0, 'failed': 0}

    def _check(name, condition, detail=""):
        if condition:
            _counts['passed'] += 1; print(f"  [PASS] {name}")
        else:
            _counts['failed'] += 1; print(f"  [FAIL] {name}  {detail}")

    # ── 1. CM determinant on known geometries ──
    print("1. Cayley-Menger determinant validation:")

    # Valid equilateral triangle in 2D
    tri_pts = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]]]).to(device)
    det_tri = cayley_menger_det(tri_pts)
    _check("equilateral triangle positive", det_tri.item() > 0,
           f"det={det_tri.item():.6f}")

    # Degenerate: collinear points
    collinear = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]).to(device)
    det_col = cayley_menger_det(collinear)
    _check("collinear points ≈ 0", abs(det_col.item()) < 1e-6,
           f"det={det_col.item():.6f}")

    # Valid tetrahedron in 3D
    tet_pts = torch.tensor([[[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0],
                              [0.5, 0.289, 0.816]]]).float().to(device)
    det_tet = cayley_menger_det(tet_pts)
    _check("tetrahedron positive", det_tet.item() > 0,
           f"det={det_tet.item():.6f}")

    # Degenerate: coplanar points in 3D
    coplanar = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                               [1, 1, 0]]]).float().to(device)
    det_cop = cayley_menger_det(coplanar)
    _check("coplanar points ≈ 0", abs(det_cop.item()) < 1e-4,
           f"det={det_cop.item():.6f}")

    # Valid pentachoron in 4D (5 vertices)
    penta = torch.randn(1, 5, 4).to(device)
    det_penta = cayley_menger_det(penta)
    _check("random pentachoron non-zero", abs(det_penta.item()) > 1e-10,
           f"det={det_penta.item():.6f}")

    # Batch: mix of valid and degenerate
    batch_pts = torch.stack([
        tri_pts[0],                                                  # valid triangle
        collinear[0],                                                # degenerate
        torch.tensor([[0, 0], [0, 1], [1, 0]]).float().to(device),  # valid right triangle
    ])
    batch_det = cayley_menger_det(batch_pts)
    _check("batch: valid/degen/valid signs",
           batch_det[0].item() > 0 and abs(batch_det[1].item()) < 1e-6 and batch_det[2].item() > 0,
           f"dets={batch_det.tolist()}")

    # ── 2. CM validity score ──
    print("\n2. CM validity score on sphere:")

    anchors = F.normalize(torch.randn(A, D, device=device), dim=-1)

    # Well-spread embeddings (should have many valid simplices)
    emb_spread = F.normalize(torch.randn(B, D, device=device), dim=-1)
    validity_spread, raw_det_spread = cm_validity_score(emb_spread, anchors, n_neighbors=3)
    pos_frac_spread = (raw_det_spread > 0).float().mean().item()
    _check("spread embeddings: shape", validity_spread.shape == (B, A))
    _check("spread embeddings: some valid", pos_frac_spread > 0.1,
           f"positive_frac={pos_frac_spread:.3f}")

    # Collapsed embeddings (all identical — should have degenerate simplices)
    emb_collapsed = F.normalize(torch.ones(B, D, device=device), dim=-1)
    validity_collapsed, raw_det_collapsed = cm_validity_score(emb_collapsed, anchors, n_neighbors=3)
    pos_frac_collapsed = (raw_det_collapsed > 0).float().mean().item()
    # Note: collapsed embeddings + spread anchors still form valid simplices
    # because the anchors provide the geometric diversity
    _check("collapsed embeddings: computable", validity_collapsed.shape == (B, A))
    print(f"       spread positive_frac:    {pos_frac_spread:.3f}")
    print(f"       collapsed positive_frac: {pos_frac_collapsed:.3f}")

    # ── 3. Anchor gate strategies ──
    print("\n3. AnchorGate strategies:")

    embedding = F.normalize(torch.randn(B, D, device=device), dim=-1)
    anchors_n = F.normalize(torch.randn(A, D, device=device), dim=-1)
    tri = 1.0 - embedding @ anchors_n.T

    for strategy in ['round_robin', 'cm_gate', 'top_k', 'top_p']:
        gate = AnchorGate(A, D, n_comp=4, n_neighbors=2,
                          strategy=strategy, top_k=8, top_p=0.8).to(device)
        gv, assign, info = gate(embedding, anchors_n, tri)
        _check(f"{strategy}: gate shape", gv.shape == (B, A))
        _check(f"{strategy}: gate range",
               gv.min().item() >= 0 and gv.max().item() <= 1.01,
               f"min={gv.min().item():.3f} max={gv.max().item():.3f}")
        if 'active' in info:
            print(f"       {strategy}: active={info['active']:.1f}")
        if 'cm_positive_frac' in info:
            print(f"       {strategy}: cm_positive={info['cm_positive_frac']:.3f}")

    # ── 4. GatedPatchwork ──
    print("\n4. GatedPatchwork forward pass:")

    for strategy in ['round_robin', 'cm_gate', 'top_k']:
        pw = GatedPatchwork(A, n_comp=4, d_comp=32, dim=D,
                            strategy=strategy, n_neighbors=2, top_k=8,
                            activation='gelu').to(device)
        out, info = pw(tri, embedding, anchors_n)
        expected_dim = 4 * 32
        _check(f"{strategy}: output shape", out.shape == (B, expected_dim),
               f"got {out.shape}")
        _check(f"{strategy}: finite", torch.isfinite(out).all().item())

    # ── 5. Gradient flow ──
    print("\n5. Gradient flow through CM gate:")

    gate = AnchorGate(A, D, n_comp=4, n_neighbors=2, strategy='cm_gate').to(device)
    emb_grad = F.normalize(torch.randn(B, D, device=device), dim=-1).requires_grad_(True)
    anc = F.normalize(torch.randn(A, D, device=device), dim=-1)
    tri_grad = (1.0 - emb_grad @ anc.T)
    gv, _, _ = gate(emb_grad, anc, tri_grad)
    loss = (gv * tri_grad).sum()
    loss.backward()
    _check("gradient reaches embedding", emb_grad.grad is not None and emb_grad.grad.abs().sum() > 0)
    _check("gradient reaches gate params",
           any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters()))

    # ── 6. Discrimination test: valid vs invalid features ──
    print("\n6. Discrimination: well-structured vs collapsed features:")

    gate = AnchorGate(A, D, n_comp=4, n_neighbors=2, strategy='cm_gate').to(device)

    # Well-structured: diverse directions on sphere
    emb_good = F.normalize(torch.randn(B, D, device=device), dim=-1)
    tri_good = 1.0 - emb_good @ anchors_n.T
    gv_good, _, info_good = gate(emb_good, anchors_n, tri_good)

    # Collapsed: all embeddings identical
    emb_bad = F.normalize(torch.ones(B, D, device=device) + 0.01 * torch.randn(B, D, device=device), dim=-1)
    tri_bad = 1.0 - emb_bad @ anchors_n.T
    gv_bad, _, info_bad = gate(emb_bad, anchors_n, tri_bad)

    good_mean = gv_good.mean().item()
    bad_mean = gv_bad.mean().item()
    good_var = gv_good.var(dim=0).mean().item()
    bad_var = gv_bad.var(dim=0).mean().item()
    print(f"       Good: gate_mean={good_mean:.4f}, gate_var={good_var:.6f}")
    print(f"       Bad:  gate_mean={bad_mean:.4f}, gate_var={bad_var:.6f}")
    _check("variance differs",
           abs(good_var - bad_var) > 1e-6 or abs(good_mean - bad_mean) > 0.01,
           f"diff_mean={abs(good_mean-bad_mean):.4f}, diff_var={abs(good_var-bad_var):.6f}")

    # ── 7. CM validity with known structure ──
    print("\n7. CM validity with controlled simplex quality:")

    # Create anchors that form a known good simplex (orthogonal in D-d)
    D_test = 8
    A_test = 5
    # Orthogonal anchors — guaranteed valid, well-conditioned simplices
    ortho_anchors = F.normalize(torch.eye(A_test, D_test, device=device), dim=-1)
    emb_test = F.normalize(torch.randn(4, D_test, device=device), dim=-1)
    validity_ortho, det_ortho = cm_validity_score(emb_test, ortho_anchors, n_neighbors=2)
    ortho_mag = det_ortho.abs().mean().item()

    # Near-duplicate anchors — technically valid but sliver simplices (tiny volume)
    dupe_base = F.normalize(torch.randn(1, D_test, device=device), dim=-1)
    dupe_anchors = dupe_base.expand(A_test, -1) + 0.001 * torch.randn(A_test, D_test, device=device)
    dupe_anchors = F.normalize(dupe_anchors, dim=-1)
    validity_dupe, det_dupe = cm_validity_score(emb_test, dupe_anchors, n_neighbors=2)
    dupe_mag = det_dupe.abs().mean().item()

    print(f"       Orthogonal anchors: |det| mean = {ortho_mag:.6f}")
    print(f"       Duplicate anchors:  |det| mean = {dupe_mag:.6f}")
    ratio = ortho_mag / (dupe_mag + 1e-15)
    print(f"       Ratio: {ratio:.1f}x")
    _check("ortho |det| >> dupe |det|", ortho_mag > dupe_mag * 10,
           f"ortho={ortho_mag:.6f} dupe={dupe_mag:.6f} ratio={ratio:.1f}x")

    # ── Summary ──
    total = _counts['passed'] + _counts['failed']
    print(f"\n{'='*50}")
    print(f"  {_counts['passed']}/{total} passed" + (f"  ({_counts['failed']} FAILED)" if _counts['failed'] else " — all clear"))
    print(f"{'='*50}")