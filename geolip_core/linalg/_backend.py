"""
Backend detection and fallback management for geolip.linalg.

Single warning on first fallback. User-configurable preferences.

Usage:
    from geolip.linalg._backend import backend

    backend.use_fl_eigh = True   # default: True when CUDA available
    backend.use_triton = True    # default: True when Triton available
    backend.warn()               # fires once, then silent
"""

import warnings
import torch

__all__ = ['backend']


class _Backend:
    """Singleton backend state for geolip.linalg."""

    def __init__(self):
        self._warned = False
        self._cuda = torch.cuda.is_available()
        self._triton = False
        self._triton_version = None

        try:
            import triton
            self._triton = True
            self._triton_version = getattr(triton, '__version__', 'unknown')
        except ImportError:
            pass

        # User-settable preferences
        self.use_fl_eigh = self._cuda
        self.use_triton = self._triton and self._cuda

    @property
    def has_cuda(self):
        return self._cuda

    @property
    def has_triton(self):
        return self._triton

    @property
    def triton_version(self):
        return self._triton_version

    def warn(self, feature: str = ''):
        """Emit a single fallback warning across all of geolip.linalg."""
        if self._warned:
            return
        self._warned = True

        parts = ['geolip.linalg: falling back to PyTorch defaults.']
        if not self._cuda:
            parts.append('CUDA not available.')
        elif not self._triton:
            parts.append('Triton not installed (pip install triton).')
        if feature:
            parts.append(f'Triggered by: {feature}.')
        parts.append(
            'FL eigh (compilable, 70/72 math purity) and Triton SVD kernels disabled. '
            'Install CUDA + Triton for full performance.'
        )
        warnings.warn(' '.join(parts), UserWarning, stacklevel=3)

    def resolve_eigh(self, A, force_fl=False):
        """Pick eigh implementation. Returns (eigenvalues, eigenvectors).

        Auto-dispatches:
          n <= 12 + CUDA + use_fl_eigh: FL pipeline (compilable, superior accuracy)
          else: torch.linalg.eigh (cuSOLVER)
        """
        n = A.shape[-1]
        if (force_fl or (self.use_fl_eigh and n <= 12)) and self._cuda and A.is_cuda:
            from .eigh import FLEigh
            return FLEigh()(A)
        if not self._cuda or not A.is_cuda:
            self.warn('eigh')
        return torch.linalg.eigh(A)

    def resolve_svd_n2(self, A, block_m=128):
        """Triton SVD for N=2, or torch fallback."""
        if self.use_triton and self._triton and A.is_cuda:
            from geolip_core.utils.kernel import batched_svd2
            return batched_svd2(A, block_m)
        if not self._triton:
            self.warn('svd2')
        return torch.linalg.svd(A.float(), full_matrices=False)

    def resolve_svd_n3(self, A, block_m=128):
        """Triton SVD for N=3, or torch fallback."""
        if self.use_triton and self._triton and A.is_cuda:
            from geolip_core.utils.kernel import batched_svd3
            return batched_svd3(A, block_m)
        if not self._triton:
            self.warn('svd3')
        return torch.linalg.svd(A.float(), full_matrices=False)

    def status(self):
        """Print backend status."""
        print(f"geolip.linalg backend:")
        print(f"  CUDA:       {'yes' if self._cuda else 'no'}")
        print(f"  Triton:     {self._triton_version if self._triton else 'not installed'}")
        print(f"  FL eigh:    {'enabled' if self.use_fl_eigh else 'disabled'}")
        print(f"  Triton SVD: {'enabled' if self.use_triton else 'disabled'}")
        if self._cuda:
            print(f"  GPU:        {torch.cuda.get_device_name()}")

    def __repr__(self):
        return (f"Backend(cuda={self._cuda}, triton={self._triton}, "
                f"fl_eigh={self.use_fl_eigh}, triton_svd={self.use_triton})")


# Module-level singleton
backend = _Backend()