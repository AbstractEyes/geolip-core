"""
Batched thin SVD with auto-dispatch and FL eigh integration.

Dispatch order:
  N=2, Triton available:  Fused Triton kernel    ~0.02ms  (fp32 only)
  N=3, Triton available:  Fused Triton kernel    ~0.02ms  (fp32 only)
  N<=12, CUDA:            Gram + FL eigh          compilable, 70/72 purity
  N>12 or CPU:            Gram + torch.linalg.eigh
  Any fallback:           torch.linalg.svd

compute_dtype controls internal precision:
  'fp64' (default): All Gram/eigh math in float64. Prevents overflow at
                    high condition numbers. ~2x slower than fp32 for eigh.
  'fp32':           Original behavior. Fast but vulnerable to conditioning.

Note: Triton kernels are fp32-only (hardware constraint). When compute_dtype='fp64'
and Triton would dispatch, falls through to Gram+eigh in fp64 instead.

Toggle FL eigh globally:
    from geolip.linalg import backend
    backend.use_fl_eigh = False  # disables FL, uses cuSOLVER everywhere

Usage:
    from geolip.linalg import svd

    U, S, Vh = svd(A)                           # auto-dispatch, fp64
    U, S, Vh = svd(A, compute_dtype='fp32')     # original fp32 behavior
    U, S, Vh = svd(A, method='fl')              # force FL eigh path
    U, S, Vh = svd(A, method='torch')           # force torch.linalg.svd
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

from .eigh import FLEigh, _FL_MAX_N
from ._backend import backend

__all__ = ['batched_svd', 'gram_fl_eigh_svd', 'gram_eigh_svd']

# Resolve dtype string to torch dtype
_DTYPE_MAP = {
    'fp64': torch.float64,
    'fp32': torch.float32,
    'float64': torch.float64,
    'float32': torch.float32,
}


def _resolve_dtype(compute_dtype):
    """Convert string or torch.dtype to torch.dtype."""
    if isinstance(compute_dtype, torch.dtype):
        return compute_dtype
    return _DTYPE_MAP.get(compute_dtype, torch.float64)


def gram_fl_eigh_svd(
    A: Tensor,
    compute_dtype: str = 'fp64',
) -> Tuple[Tensor, Tensor, Tensor]:
    """Thin SVD via Gram matrix + FL eigendecomposition.

    Fully compilable for n <= 12. Zero graph breaks.

    Args:
        A: (B, M, N) tensor, M >= N, N <= 12
        compute_dtype: 'fp64' or 'fp32' for internal computation

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) -- singular values descending.
             Output dtype matches input dtype.
    """
    orig_dtype = A.dtype
    dt = _resolve_dtype(compute_dtype)

    with torch.amp.autocast('cuda', enabled=False):
        A_c = A.to(dt)
        G = torch.bmm(A_c.transpose(1, 2), A_c)

        # FLEigh operates in fp64 internally for polynomial phase,
        # but expects fp32 input for Newton-Schulz. Cast G appropriately.
        if dt == torch.float64:
            # Run eigh in fp64, FL's internal fp64 phases align naturally
            eigenvalues, V = FLEigh()(G.float())  # FL needs float32 input
            eigenvalues = eigenvalues.to(dt)
            V = V.to(dt)
        else:
            eigenvalues, V = FLEigh()(G)

        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-24 if dt == torch.float64 else 1e-12))
        U = torch.bmm(A_c, V) / S.unsqueeze(1).clamp(min=1e-16 if dt == torch.float64 else 1e-8)
        Vh = V.transpose(-2, -1).contiguous()

    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)


def gram_eigh_svd(
    A: Tensor,
    compute_dtype: str = 'fp64',
) -> Tuple[Tensor, Tensor, Tensor]:
    """Thin SVD via Gram matrix + torch.linalg.eigh.

    Fallback for N > 12 or when FL is disabled.

    Args:
        A: (B, M, N) tensor, M >= N
        compute_dtype: 'fp64' or 'fp32' for internal computation

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) -- singular values descending.
             Output dtype matches input dtype.
    """
    orig_dtype = A.dtype
    dt = _resolve_dtype(compute_dtype)
    clamp_min = 1e-24 if dt == torch.float64 else 1e-12
    div_min = 1e-16 if dt == torch.float64 else 1e-8

    with torch.amp.autocast('cuda', enabled=False):
        A_c = A.to(dt)
        G = torch.bmm(A_c.transpose(1, 2), A_c)
        eigenvalues, V = torch.linalg.eigh(G)
        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=clamp_min))
        U = torch.bmm(A_c, V) / S.unsqueeze(1).clamp(min=div_min)
        Vh = V.transpose(-2, -1).contiguous()

    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)


def batched_svd(
    A: Tensor,
    method: str = 'auto',
    block_m: int = 128,
    compute_dtype: str = 'fp64',
) -> Tuple[Tensor, Tensor, Tensor]:
    """Batched thin SVD for (B, M, N) tensors. M >= N.

    Methods:
      'auto':       Best available for each N (respects compute_dtype)
      'fl':         Force Gram + FL eigh (N <= 12)
      'gram_eigh':  Force Gram + torch.linalg.eigh
      'triton':     Force Triton kernel (N=2,3 only, fp32 only)
      'torch':      Force torch.linalg.svd

    Args:
        A:             (B, M, N) tensor
        method:        Dispatch method
        block_m:       Tile size for Triton kernels
        compute_dtype: 'fp64' (default) or 'fp32' for internal precision

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) -- singular values descending.
             Output dtype matches input dtype.
    """
    assert A.ndim == 3, f"Expected (B, M, N), got {A.shape}"
    B, M, N = A.shape
    assert M >= N, f"Thin SVD requires M >= N, got M={M}, N={N}"

    dt = _resolve_dtype(compute_dtype)
    use_fp64 = (dt == torch.float64)
    orig_dtype = A.dtype

    if method == 'torch':
        U, S, Vh = torch.linalg.svd(A.to(dt), full_matrices=False)
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)

    if method == 'fl':
        return gram_fl_eigh_svd(A, compute_dtype=compute_dtype)

    if method == 'gram_eigh':
        return gram_eigh_svd(A, compute_dtype=compute_dtype)

    if method == 'triton':
        if use_fp64:
            # Triton kernels are fp32-only, fall through to gram_eigh in fp64
            return gram_eigh_svd(A, compute_dtype=compute_dtype)
        if N == 2:
            return backend.resolve_svd_n2(A, block_m)
        elif N == 3:
            return backend.resolve_svd_n3(A, block_m)
        raise ValueError(f"Triton kernel only for N=2,3, got N={N}")

    # method == 'auto'
    # When fp64 requested, skip Triton (fp32-only) and use Gram+eigh in fp64
    if not use_fp64 and N == 2 and backend.use_triton and A.is_cuda:
        return backend.resolve_svd_n2(A, block_m)
    elif not use_fp64 and N == 3 and backend.use_triton and A.is_cuda:
        return backend.resolve_svd_n3(A, block_m)
    elif N <= _FL_MAX_N and backend.use_fl_eigh and A.is_cuda:
        return gram_fl_eigh_svd(A, compute_dtype=compute_dtype)
    elif A.is_cuda:
        return gram_eigh_svd(A, compute_dtype=compute_dtype)
    else:
        backend.warn('svd')
        U, S, Vh = torch.linalg.svd(A.to(dt), full_matrices=False)
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)