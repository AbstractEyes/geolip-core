"""
geolip.linalg — Geometric linear algebra primitives.

Compilable eigendecomposition, batched SVD, Procrustes alignment.

Quick start:
    from geolip.linalg import eigh, svd, procrustes

    # Eigendecomposition (FL Hybrid, 70/72 math purity vs cuSOLVER)
    eigenvalues, eigenvectors = eigh(A)

    # For torch.compile:
    from geolip.linalg import FLEigh
    solver = torch.compile(FLEigh(), fullgraph=True)

    # Batched thin SVD (auto-dispatches N=2,3 to Triton, N<=12 to FL)
    U, S, Vh = svd(A)

    # Procrustes alignment
    aligned, info = procrustes(source, target)
"""

from .eigh import FLEigh, fl_eigh, eigh
from .svd import batched_svd, gram_eigh_svd, gram_fl_eigh_svd
from .newton_schulz import newton_schulz_invsqrt
from .procrustes import batched_procrustes

# Convenience aliases
svd = batched_svd
procrustes = batched_procrustes

__all__ = [
    # Eigendecomposition
    'FLEigh',
    'fl_eigh',
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