"""
geolip.linalg — Geometric linear algebra primitives.

Compilable eigendecomposition, batched SVD, Procrustes alignment.
Falls back to PyTorch defaults with a single warning when CUDA/Triton unavailable.

Quick start::

    from geolip.linalg import eigh, svd, procrustes

    vals, vecs = eigh(A)                     # FL for n<=12, cuSOLVER otherwise
    U, S, Vh = svd(A)                        # Triton N=2,3 -> FL N<=12 -> cuSOLVER
    aligned, info = procrustes(src, tgt)

Configuration::

    from geolip.linalg import backend

    backend.status()                          # print what's available
    backend.use_fl_eigh = False               # disable FL, use cuSOLVER everywhere
    backend.use_triton = False                # disable Triton SVD kernels

For torch.compile::

    from geolip.linalg import FLEigh
    solver = torch.compile(FLEigh(), fullgraph=True)
    vals, vecs = solver(A)                    # zero graph breaks
"""

from ._backend import backend
from .eigh import FLEigh, eigh
from .svd import batched_svd, gram_eigh_svd, gram_fl_eigh_svd
from .newton_schulz import newton_schulz_invsqrt
from .procrustes import batched_procrustes

# Convenience aliases
svd = batched_svd
procrustes = batched_procrustes

__all__ = [
    # Backend
    'backend',
    # Eigendecomposition
    'FLEigh',
    'eigh',
    # SVD
    'batched_svd',
    'gram_eigh_svd',
    'gram_fl_eigh_svd',
    'svd',
    # Newton-Schulz
    'newton_schulz_invsqrt',
    # Procrustes
    'batched_procrustes',
    'procrustes',
]