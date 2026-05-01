"""
Batched thin SVD with auto-dispatch and FL eigh integration.

Dispatch order:
  N=2..8, Triton + CUDA:        Fused Triton kernel  (fp32 or fp64)
  N<=12, CUDA, fp32:            Gram + FL eigh       (compilable, 70/72 purity)
  N<=12, CUDA, fp64:            Gram + torch.linalg.eigh
                                (FLEigh returns fp32 V; using torch.linalg.eigh
                                 keeps the precision the user asked for.)
  N>12 or CPU:                  Gram + torch.linalg.eigh
  Any fallback:                 torch.linalg.svd

Wide shapes (M < N) are handled transparently: the input is transposed,
the standard thin SVD is computed on the (now tall) transpose, and U / Vh
are swapped on return. So `batched_svd((B, 2, 3))` is valid and routes
through the N=2 Triton kernel.

compute_dtype controls internal precision:
  'fp64' (default): All math in float64. Prevents overflow at high
                    condition numbers. ~2x slower than fp32.
  'fp32':           Faster, but vulnerable to ill-conditioning.

Triton kernels honor both fp32 and fp64 (DTYPE is a kernel constexpr).

Toggle FL eigh / Triton globally:
    from geolip_core.linalg import backend
    backend.use_fl_eigh = False  # disables FL, uses cuSOLVER everywhere
    backend.use_triton  = False  # disables Triton kernels everywhere

Usage:
    from geolip_core.linalg import svd

    U, S, Vh = svd(A)                           # auto-dispatch, fp64
    U, S, Vh = svd(A, compute_dtype='fp32')     # fp32 fast path
    U, S, Vh = svd(A, method='fl')              # force FL eigh path
    U, S, Vh = svd(A, method='triton')          # force Triton (N=2..8)
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


_TRITON_RESOLVERS = None  # populated lazily to avoid import-time circularity


def _triton_resolver(N):
    global _TRITON_RESOLVERS
    if _TRITON_RESOLVERS is None:
        _TRITON_RESOLVERS = {
            2: backend.resolve_svd_n2,
            3: backend.resolve_svd_n3,
            4: backend.resolve_svd_n4,
            5: backend.resolve_svd_n5,
            6: backend.resolve_svd_n6,
            7: backend.resolve_svd_n7,
            8: backend.resolve_svd_n8,
        }
    return _TRITON_RESOLVERS.get(N)


def batched_svd(
    A: Tensor,
    method: str = 'auto',
    block_m: int = 128,
    compute_dtype: str = 'fp64',
) -> Tuple[Tensor, Tensor, Tensor]:
    """Batched thin SVD for (B, M, N) tensors. Wide shapes (M < N) are
    transparently handled by transposing through the tall path.

    Methods:
      'auto':       Best available for each N (respects compute_dtype)
      'fl':         Force Gram + FL eigh (N <= 12)
      'gram_eigh':  Force Gram + torch.linalg.eigh
      'triton':     Force Triton kernel (N=2..8, fp32 or fp64)
      'torch':      Force torch.linalg.svd

    Args:
        A:             (B, M, N) tensor
        method:        Dispatch method
        block_m:       Tile size for Triton kernels
        compute_dtype: 'fp64' (default) or 'fp32' for internal precision

    Returns: U (B,M,N), S (B,K), Vh (B,K,N) where K = min(M, N) --
             singular values descending. Output dtype matches input dtype.
    """
    assert A.ndim == 3, f"Expected (B, M, N), got {A.shape}"
    B, M, N = A.shape
    assert M >= 1 and N >= 1, f"Empty matrix: M={M}, N={N}"

    # Wide-shape shim: thin SVD requires M >= N; transpose, recurse, swap.
    # A = U S Vh  <=>  A^T = (V S U^T) = (Vh^T) S (U^T)
    if M < N:
        A_t = A.transpose(-1, -2).contiguous()
        Ut, S, Vht = batched_svd(A_t, method=method, block_m=block_m,
                                 compute_dtype=compute_dtype)
        return Vht.transpose(-1, -2), S, Ut.transpose(-1, -2)

    dt = _resolve_dtype(compute_dtype)
    orig_dtype = A.dtype

    if method == 'torch':
        U, S, Vh = torch.linalg.svd(A.to(dt), full_matrices=False)
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)

    if method == 'fl':
        return gram_fl_eigh_svd(A, compute_dtype=compute_dtype)

    if method == 'gram_eigh':
        return gram_eigh_svd(A, compute_dtype=compute_dtype)

    if method == 'triton':
        resolver = _triton_resolver(N)
        if resolver is None:
            raise ValueError(f"Triton kernel only for N=2..8, got N={N}")
        A_c = A.to(dt) if A.dtype != dt else A
        U, S, Vh = resolver(A_c, block_m)
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)

    # method == 'auto'
    use_fp64 = (dt == torch.float64)
    if 2 <= N <= 8 and backend.use_triton and A.is_cuda:
        A_c = A.to(dt) if A.dtype != dt else A
        U, S, Vh = _triton_resolver(N)(A_c, block_m)
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
    # FLEigh internally returns fp32 eigenvectors (eigh.py:135 `vec.float()`),
    # which silently caps fp64 SVD orthogonality at ~1e-3 even with fp64 input.
    # Skip FL when compute_dtype is fp64 — torch.linalg.eigh stays in fp64.
    elif N <= _FL_MAX_N and backend.use_fl_eigh and A.is_cuda and not use_fp64:
        return gram_fl_eigh_svd(A, compute_dtype=compute_dtype)
    elif A.is_cuda:
        return gram_eigh_svd(A, compute_dtype=compute_dtype)
    else:
        backend.warn('svd')
        U, S, Vh = torch.linalg.svd(A.to(dt), full_matrices=False)
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)