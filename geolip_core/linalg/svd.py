"""
Batched thin SVD with auto-dispatch and FL eigh integration.

Dispatch order:
  N=2, Triton available:  Fused Triton kernel    ~0.02ms
  N=3, Triton available:  Fused Triton kernel    ~0.02ms
  N<=12, CUDA:            Gram + FL eigh          compilable, 70/72 purity
  N>12 or CPU:            Gram + torch.linalg.eigh
  Any fallback:           torch.linalg.svd

Toggle FL eigh globally:
    from geolip.linalg import backend
    backend.use_fl_eigh = False  # disables FL, uses cuSOLVER everywhere

Usage:
    from geolip.linalg import svd

    U, S, Vh = svd(A)                           # auto-dispatch
    U, S, Vh = svd(A, method='fl')              # force FL eigh path
    U, S, Vh = svd(A, method='torch')           # force torch.linalg.svd
"""

import torch
from torch import Tensor
from typing import Tuple

from .eigh import FLEigh, _FL_MAX_N
from ._backend import backend

__all__ = ['batched_svd', 'gram_fl_eigh_svd', 'gram_eigh_svd']


def gram_fl_eigh_svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Thin SVD via Gram matrix + FL eigendecomposition.

    Fully compilable for n <= 12. Zero graph breaks.

    Args:
        A: (B, M, N) tensor, M >= N, N <= 12

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    with torch.amp.autocast('cuda', enabled=False):
        A_f = A.float()
        G = torch.bmm(A_f.transpose(1, 2), A_f)
        eigenvalues, V = FLEigh()(G)
        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-12))
        U = torch.bmm(A_f, V) / S.unsqueeze(1)
        Vh = V.transpose(-2, -1).contiguous()
    return U, S, Vh


def gram_eigh_svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Thin SVD via Gram matrix + torch.linalg.eigh.

    Fallback for N > 12 or when FL is disabled.

    Args:
        A: (B, M, N) tensor, M >= N

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
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

    Methods:
      'auto':       Best available for each N
      'fl':         Force Gram + FL eigh (N <= 12)
      'gram_eigh':  Force Gram + torch.linalg.eigh
      'triton':     Force Triton kernel (N=2,3 only)
      'torch':      Force torch.linalg.svd

    Args:
        A:        (B, M, N) tensor
        method:   Dispatch method
        block_m:  Tile size for Triton kernels

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    assert A.ndim == 3, f"Expected (B, M, N), got {A.shape}"
    B, M, N = A.shape
    assert M >= N, f"Thin SVD requires M >= N, got M={M}, N={N}"

    if method == 'torch':
        return torch.linalg.svd(A.float(), full_matrices=False)

    if method == 'fl':
        return gram_fl_eigh_svd(A)

    if method == 'gram_eigh':
        return gram_eigh_svd(A)

    if method == 'triton':
        if N == 2:
            return backend.resolve_svd_n2(A, block_m)
        elif N == 3:
            return backend.resolve_svd_n3(A, block_m)
        raise ValueError(f"Triton kernel only for N=2,3, got N={N}")

    # method == 'auto'
    if N == 2 and backend.use_triton and A.is_cuda:
        return backend.resolve_svd_n2(A, block_m)
    elif N == 3 and backend.use_triton and A.is_cuda:
        return backend.resolve_svd_n3(A, block_m)
    elif N <= _FL_MAX_N and backend.use_fl_eigh and A.is_cuda:
        return gram_fl_eigh_svd(A)
    elif A.is_cuda:
        return gram_eigh_svd(A)
    else:
        backend.warn('svd')
        return torch.linalg.svd(A.float(), full_matrices=False)