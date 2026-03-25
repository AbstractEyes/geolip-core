"""
utils — Engineering infrastructure.

Triton kernels, linear algebra primitives, memory management.
These make the geometric behaviors fast and numerically stable.
The behaviors in core/ call these; they don't own the math.
"""

from .kernel import (
    batched_svd, batched_svd2, batched_svd3,
    gram_eigh_svd, newton_schulz_invsqrt,
    batched_procrustes, HAS_TRITON,
)
from .memory import EmbeddingBuffer
