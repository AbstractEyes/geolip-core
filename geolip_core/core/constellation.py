"""
Constellation — learned anchors on S^(d-1) for triangulation.

The core geometric primitive. Every input is located by its cosine
distances to all anchors. The triangulation IS the representation.

Usage:
    from core.constellation import Constellation, init_anchors_repulsion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ANCHOR INITIALIZATION ──

def init_anchors_xavier(n, d):
    """Xavier-normal init, L2-normalized."""
    w = torch.empty(n, d)
    nn.init.xavier_normal_(w)
    return F.normalize(w, dim=-1)


def init_anchors_orthogonal(n, d):
    """QR decomposition for maximal initial spread.

    If n <= d: returns orthonormal rows.
    If n > d: orthonormal base + random normalized extras.
    """
    if n <= d:
        Q, _ = torch.linalg.qr(torch.randn(d, n))
        return Q.T.contiguous()
    else:
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        return torch.cat([Q.T, F.normalize(torch.randn(n - d, d), dim=-1)], dim=0)


def init_anchors_repulsion(n, d, iters=200, lr=0.05):
    """Orthogonal init + iterative repulsion for uniform coverage.

    Starts from orthogonal placement, then pushes closest pairs apart.
    200 iterations is sufficient for convergence.
    """
    vecs = F.normalize(init_anchors_orthogonal(n, d), dim=-1)
    for _ in range(iters):
        sim = vecs @ vecs.T
        sim.fill_diagonal_(-2.0)
        vecs = F.normalize(vecs - lr * vecs[sim.argmax(dim=1)], dim=-1)
    return vecs


INIT_METHODS = {
    'xavier': init_anchors_xavier,
    'orthogonal': init_anchors_orthogonal,
    'repulsion': init_anchors_repulsion,
}


# ── CONSTELLATION ──

class Constellation(nn.Module):
    """Learned anchors on S^(d-1). Triangulates input embeddings.

    Triangulation: emb → cosine distance to all anchors → (B, n_anchors)

    The triangulation profile is the geometric fingerprint of each input.
    Two embeddings with identical distances to all anchors are
    indistinguishable — the anchor frame IS the representation.

    Args:
        n_anchors: number of anchor points
        dim: embedding dimension
        anchor_drop: probability of dropping each anchor during training
        anchor_init: initialization strategy ('repulsion', 'orthogonal', 'xavier')
    """

    def __init__(self, n_anchors, dim, anchor_drop=0.0, anchor_init='repulsion'):
        super().__init__()
        self.anchors = nn.Parameter(INIT_METHODS[anchor_init](n_anchors, dim))
        self.anchor_drop = anchor_drop
        self.n_anchors = n_anchors
        self.dim = dim

    def triangulate(self, emb, training=False):
        """Compute distance profile from emb to all anchors.

        Args:
            emb: (B, D) L2-normalized embeddings on S^(d-1)
            training: if True, apply anchor dropout

        Returns:
            tri: (B, A) cosine distances (1 - cos) to each anchor
            nearest: (B,) index of closest anchor
        """
        anchors = F.normalize(self.anchors, dim=-1)
        if training and self.anchor_drop > 0:
            mask = torch.rand(anchors.shape[0], device=anchors.device) > self.anchor_drop
            if mask.sum() < 2:
                mask[:2] = True
            anchors = anchors[mask]
            cos = emb @ anchors.T
            tri = 1.0 - cos
            _, nl = cos.max(dim=-1)
            return tri, mask.nonzero(as_tuple=True)[0][nl]
        cos = emb @ anchors.T
        tri = 1.0 - cos
        _, nearest = cos.max(dim=-1)
        return tri, nearest

    def forward(self, emb, training=False):
        return self.triangulate(emb, training)