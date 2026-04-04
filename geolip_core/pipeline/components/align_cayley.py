"""
CayleyOrthogonal — guaranteed SO(d) rotation via Cayley map.
Proven in Ryan Spearman (rho=0.309, 76/84 wins).

CRITICAL: precompute() MUST have @torch.no_grad() to prevent
autograd graph linking through .copy_(). See CLAUDE.md rules.
"""

import torch
import torch.nn as nn
import geolip_core.linalg as LA
from geolip_core.pipeline.observer import TorchComponent


class CayleyOrthogonal(TorchComponent):
    """Guaranteed SO(d) rotation via Cayley map. det(Q) = 1 always.

    Cache uses a pre-allocated registered buffer updated in-place.
    CUDA graph replay requires fixed tensor addresses — allocating new
    tensors on each precompute would invalidate recorded addresses.
    """
    def __init__(self, name, dim):
        super().__init__(name)
        self.dim = dim
        self.A_upper = nn.Parameter(torch.zeros(dim * (dim - 1) // 2) * 0.01)
        idx = torch.triu_indices(dim, dim, offset=1)
        self.register_buffer('_triu_row', idx[0], persistent=False)
        self.register_buffer('_triu_col', idx[1], persistent=False)
        self.register_buffer('_eye', torch.eye(dim), persistent=False)
        # Pre-allocated cache — address-stable for CUDA graph replay
        self.register_buffer('_cached_rotation', torch.eye(dim), persistent=False)
        self._cache_warm = False

    def get_rotation(self):
        """Compute rotation via Cayley map. Uses cuSOLVER (LA.solve)."""
        d = self.dim
        A = torch.zeros(d, d, device=self.A_upper.device, dtype=self.A_upper.dtype)
        A[self._triu_row, self._triu_col] = self.A_upper
        A = A - A.T
        return LA.solve(self._eye + A, self._eye - A)

    @torch.no_grad()
    def precompute(self):
        """Update cached rotation in-place. Call outside compiled graph."""
        self._cached_rotation.copy_(self.get_rotation())
        self._cache_warm = True

    def invalidate_cache(self):
        self._cache_warm = False

    def forward(self, x):
        if not self._cache_warm:
            with torch.no_grad():
                self._cached_rotation.copy_(self.get_rotation())
            self._cache_warm = True
        return x @ self._cached_rotation.T
