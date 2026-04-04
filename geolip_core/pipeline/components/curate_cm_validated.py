"""
CM Validated Gate — efficient anchor gating for transformer scale.

Precomputes anchor CM quality O(A²) and caches it, then combines
with per-position proximity features through a learned gate.
"""

import torch
import torch.nn as nn
import geolip_core.linalg as LA


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

        # Pre-allocated cache — address-stable for CUDA graph replay.
        # .copy_() updates values without changing tensor address.
        self.register_buffer('_cached_cm_norm', torch.zeros(n_anchors), persistent=False)
        self._cache_warm = False

    def invalidate_cache(self):
        """Mark cache stale. Buffer stays allocated (same address)."""
        self._cache_warm = False

    def precompute(self, anchors):
        """Compute anchor CM norm OUTSIDE compile graph.
        Updates buffer in-place via .copy_() — address stays fixed.
        """
        if self._cache_warm:
            return
        with torch.no_grad():
            anchor_cm, _ = anchor_neighborhood_cm(anchors, self.n_neighbors)
            cm_std = anchor_cm.std().clamp(min=1e-8)
            new_val = ((anchor_cm - anchor_cm.mean()) / cm_std).detach()
            self._cached_cm_norm.copy_(new_val)
            self._cache_warm = True

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
