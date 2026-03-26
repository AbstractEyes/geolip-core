"""
CurateMagnitude — Curate stage TorchComponent.

Thin cache adapter around core.curate.patchwork.MagnitudeFlow.

Reads: embedding, tri, magnitude
Writes: mag_anchors, mag_comp
"""

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.curate.patchwork import MagnitudeFlow


class CurateMagnitude(TorchComponent):
    """Relay-stack magnitude prediction. Wraps MagnitudeFlow.

    Reads:  cache['embedding']  — (B, D)
            cache['tri']        — (B, A)
            cache['magnitude']  — (B, 1) raw magnitude
    Writes: cache['mag_anchors'] — (B, A) per-anchor magnitude
            cache['mag_comp']    — (B, n_comp) per-compartment magnitude
    """

    def __init__(self, name, dim, n_anchors, hidden_dim=64,
                 n_layers=2, n_comp=8, **kwargs):
        super().__init__(name, **kwargs)
        self.inner = MagnitudeFlow(
            dim, n_anchors, hidden_dim, n_layers=n_layers, n_comp=n_comp)

    def forward(self, emb=None, tri=None, magnitude=None):
        if self.parent is not None:
            if emb is None:
                emb = self.parent.cache_get('embedding')
            if tri is None:
                tri = self.parent.cache_get('tri')
            if magnitude is None:
                magnitude = self.parent.cache_get('magnitude')

        mag, mag_comp = self.inner(emb, tri, magnitude)

        if self.parent is not None:
            self.parent.cache_set('mag_anchors', mag)
            self.parent.cache_set('mag_comp', mag_comp)
        return mag, mag_comp
