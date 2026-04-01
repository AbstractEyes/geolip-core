"""
Batched thin SVD with auto-dispatch.

Uses FL eigendecomposition for the Gram-eigh path (n <= 12),
with Triton fused kernels for N=2,3.

Usage:
    from geolip.linalg import svd

    U, S, Vh = svd(A)  # auto-dispatches by N
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

from .eigh import FLEigh, _FL_MAX_N

__all__ = ['batched_svd', 'gram_eigh_svd', 'gram_fl_eigh_svd']

# Triton kernels imported from kernel.py if available
HAS_TRITON = False
_svd2_fn = None
_svd3_fn = None

try:
    from geolip.kernel import batched_svd2, batched_svd3, HAS_TRITON as _HT
    HAS_TRITON = _HT
    _svd2_fn = batched_svd2
    _svd3_fn = batched_svd3
except ImportError:
    pass


def gram_fl_eigh_svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Thin SVD via Gram matrix + FL eigendecomposition.

    G = A^T A -> FL_eigh(G) -> S = sqrt(eigenvalues), V = eigenvectors, U = AV/S

    Fully compilable for n <= 12. Zero graph breaks.

    Args:
        A: (B, M, N) tensor, M >= N, N <= 12

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    B, M, N = A.shape
    with torch.amp.autocast('cuda', enabled=False):
        A_f = A.float()
        G = torch.bmm(A_f.transpose(1, 2), A_f)
        eigenvalues, V = FLEigh()(G)
        # FL returns ascending; flip to descending
        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-12))
        U = torch.bmm(A_f, V) / S.unsqueeze(1)
        Vh = V.transpose(-2, -1).contiguous()
    return U, S, Vh


def gram_eigh_svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Thin SVD via Gram matrix + torch.linalg.eigh.

    Fallback for N > 12 where FL is not optimal.

    Args:
        A: (B, M, N) tensor, M >= N

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    B, M, N = A.shape
    with torch.amp.autocast('cuda', enabled=False):
        A_f = A.float()
        G = torch.bmm(A_f.transpose(1, 2), A_f)
        eigenvalues, V = torch.linalg.eigh(G)
        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-12))
        U = torch.bmm(A_f, V) / S.unsqueeze(1)
        Vh = V.transpose(-2, -1).contiguous()
    return U, S, Vh


def batched_svd(
    A: Tensor,
    method: str = 'auto',
    block_m: int = 128,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Batched thin SVD for (B, M, N) tensors. M >= N.

    Auto-dispatches by N:
      N=2:    Fused Triton kernel     ~0.02ms
      N=3:    Fused Triton kernel     ~0.02ms
      N<=12:  Gram + FL eigh          compilable, 70/72 math purity
      N>12:   Gram + cuSOLVER eigh    standard path

    Args:
        A:        (B, M, N) tensor
        method:   'auto', 'fl', 'gram_eigh', 'triton', 'torch'
        block_m:  Tile size for Triton kernels

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    assert A.ndim == 3, f"Expected (B, M, N), got {A.shape}"
    B, M, N = A.shape
    assert M >= N, f"Thin SVD requires M >= N, got M={M}, N={N}"

    if method == 'auto':
        if N == 2 and HAS_TRITON and _svd2_fn and A.is_cuda:
            return _svd2_fn(A, block_m)
        elif N == 3 and HAS_TRITON and _svd3_fn and A.is_cuda:
            return _svd3_fn(A, block_m)
        elif N <= _FL_MAX_N:
            return gram_fl_eigh_svd(A)
        else:
            return gram_eigh_svd(A)
    elif method == 'fl':
        return gram_fl_eigh_svd(A)
    elif method == 'gram_eigh':
        return gram_eigh_svd(A)
    elif method == 'triton':
        if N == 2 and _svd2_fn:
            return _svd2_fn(A, block_m)
        elif N == 3 and _svd3_fn:
            return _svd3_fn(A, block_m)
        raise ValueError(f"Triton kernel only for N=2,3, got N={N}")
    elif method == 'torch':
        return torch.linalg.svd(A.float(), full_matrices=False)
    raise ValueError(f"Unknown method '{method}'")