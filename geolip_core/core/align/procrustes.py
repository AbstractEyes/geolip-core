"""
Procrustes alignment — subspace-preserving rotation between embedding spaces.

Wraps the mathematical primitives in utils/kernel.py into a reusable
alignment module with state tracking and diagnostics.

Usage:
    from geolip_core.core.align.procrustes import ProcrustesAlignment

    aligner = ProcrustesAlignment(dim=384, rank=24)
    aligned, info = aligner(source_embeddings, target_embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.utils.kernel import batched_procrustes, newton_schulz_invsqrt


class ProcrustesAlignment(nn.Module):
    """Subspace-preserving Procrustes alignment.

    N ≤ 32: full Procrustes.
    N > 32: project to rank-d, align, lift back preserving
            orthogonal complement exactly.

    Validated: 1.000 NN agreement with full Procrustes (N=32-128, k=8-64).

    Args:
        dim:    Embedding dimension
        rank:   Projection rank for dim > 32
        whiten: Apply Newton-Schulz whitening
    """

    def __init__(self, dim=384, rank=24, whiten=True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.whiten = whiten

        # Track alignment history
        self.register_buffer('n_alignments', torch.tensor(0, dtype=torch.long))
        self.register_buffer('last_cos_after', torch.tensor(0.0))

    def forward(self, source, target):
        """Align source to target.

        Args:
            source: (B, n_samples, N) or (n_samples, N)
            target: (B, n_samples, N) or (n_samples, N)

        Returns:
            aligned: same shape as source
            info:    dict with method, rotation, diagnostics
        """
        aligned, info = batched_procrustes(
            source, target, rank=self.rank, whiten=self.whiten)

        with torch.no_grad():
            self.n_alignments += 1
            self.last_cos_after.fill_(info['cos_after'])

        return aligned, info

    def diagnostics(self):
        """Return alignment health summary."""
        return {
            'n_alignments': self.n_alignments.item(),
            'last_cos_after': self.last_cos_after.item(),
            'rank': self.rank,
            'whiten': self.whiten,
        }
