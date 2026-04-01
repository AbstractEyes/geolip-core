"""
Newton-Schulz iterative inverse square root.

Pure bmm — zero eigensolvers. Quadratic convergence.
"""

import torch
from torch import Tensor

__all__ = ['newton_schulz_invsqrt']


def newton_schulz_invsqrt(G: Tensor, iters: int = 10) -> Tensor:
    """Batched G^{-1/2} via Newton-Schulz iteration.

    Args:
        G:     (B, N, N) symmetric PSD matrices
        iters: Iteration count (10 conservative, 7 usually sufficient)

    Returns: (B, N, N) inverse square root matrices
    """
    B, N, _ = G.shape
    device = G.device
    with torch.amp.autocast('cuda', enabled=False):
        G = G.float()
        trace = G.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
        G_norm = G / trace
        I = torch.eye(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        Y = G_norm.clone()
        Z = I.clone()
        for _ in range(iters):
            ZY = torch.bmm(Z, Y)
            factor = 1.5 * I - 0.5 * ZY
            Y = torch.bmm(Y, factor)
            Z = torch.bmm(factor, Z)
        return Z / trace.sqrt()