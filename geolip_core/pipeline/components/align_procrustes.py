"""
AlignProcrustes — Align stage TorchComponent.

Thin cache adapter around core.align.procrustes.ProcrustesAlignment.

Writes: aligned, alignment_info
"""

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.align.procrustes import ProcrustesAlignment


class AlignProcrustes(TorchComponent):
    """Subspace-preserving Procrustes alignment. Wraps ProcrustesAlignment.

    Reads:  (direct inputs or cache)
    Writes: cache['aligned']        — aligned source embeddings
            cache['alignment_info'] — dict with method, cos_before, cos_after
    """

    def __init__(self, name, dim=384, rank=24, whiten=True, **kwargs):
        super().__init__(name, **kwargs)
        self.inner = ProcrustesAlignment(dim, rank, whiten)

    def forward(self, source=None, target=None):
        aligned, info = self.inner(source, target)
        if self.parent is not None:
            self.parent.cache_set('aligned', aligned)
            self.parent.cache_set('alignment_info', info)
        return aligned, info
