"""
geolip.linalg — Drop-in replacement for torch.linalg.

Our implementations override where we have something better.
Everything else transparently proxies to torch.linalg.

    import geolip.linalg as LA

    vals, vecs = LA.eigh(A)          # FL pipeline (n<=12), cuSOLVER otherwise
    U, S, Vh = LA.svd(A)             # Triton/FL/cuSOLVER auto-dispatch
    x = LA.solve(A, b)               # passthrough to torch.linalg.solve
    L = LA.cholesky(A)               # passthrough to torch.linalg.cholesky
    n = LA.norm(x)                   # passthrough to torch.linalg.norm

Configuration::

    from geolip.linalg import backend
    backend.status()                  # show what's available
    backend.use_fl_eigh = False       # disable FL, cuSOLVER everywhere
"""

import torch.linalg as _torch_linalg

from ._backend import backend
from .eigh import FLEigh, eigh
from .conduit import FLEighConduit, ConduitPacket, canonicalize_eigenvectors, verify_parity
from .svd import batched_svd, gram_eigh_svd, gram_fl_eigh_svd
from .newton_schulz import newton_schulz_invsqrt
from .procrustes import batched_procrustes

# Our overrides — these names shadow torch.linalg when accessed via geolip.linalg
svd = batched_svd
procrustes = batched_procrustes

__all__ = [
    # Backend
    'backend',
    # Eigendecomposition (ours)
    'FLEigh',
    'eigh',
    # Conduit — evidence-emitting eigendecomposition
    'FLEighConduit',
    'ConduitPacket',
    'canonicalize_eigenvectors',
    'verify_parity',
    # SVD (ours)
    'batched_svd',
    'gram_eigh_svd',
    'gram_fl_eigh_svd',
    'svd',
    # Newton-Schulz (ours)
    'newton_schulz_invsqrt',
    # Procrustes (ours)
    'batched_procrustes',
    'procrustes',
]


def __getattr__(name):
    """Proxy anything we haven't overridden to torch.linalg.

    This makes geolip.linalg a superset of torch.linalg:
      - geolip.linalg.eigh     -> our FL pipeline
      - geolip.linalg.solve    -> torch.linalg.solve
      - geolip.linalg.cholesky -> torch.linalg.cholesky
      - etc.
    """
    if hasattr(_torch_linalg, name):
        return getattr(_torch_linalg, name)
    raise AttributeError(f"module 'geolip.linalg' has no attribute '{name}'")