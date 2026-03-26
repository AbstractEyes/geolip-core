"""
CuratePatchwork / CurateGatedPatchwork — Curate stage TorchComponents.

Thin cache adapters around core.curate.patchwork.Patchwork
and core.curate.gate.GatedPatchwork.

Writes: patchwork, gate_info (gated variant)
"""

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.curate.patchwork import Patchwork
from geolip_core.core.curate.gate import GatedPatchwork


class CuratePatchwork(TorchComponent):
    """Round-robin compartmentalized interpretation. Wraps Patchwork.

    Reads:  cache['tri_gated'] or cache['tri']
    Writes: cache['patchwork'] — (B, n_comp * d_comp)
    """

    def __init__(self, name, n_anchors, n_comp=8, d_comp=64,
                 activation='squared_relu', **kwargs):
        super().__init__(name, **kwargs)
        self.inner = Patchwork(n_anchors, n_comp, d_comp, activation)

    def forward(self, tri=None):
        if tri is None and self.parent is not None:
            tri = self.parent.cache_get('tri_gated')
            if tri is None:
                tri = self.parent.cache_get('tri')

        pw = self.inner(tri)

        if self.parent is not None:
            self.parent.cache_set('patchwork', pw)
        return pw

    @property
    def output_dim(self):
        return self.inner.output_dim


class CurateGatedPatchwork(TorchComponent):
    """CM-gated patchwork (gate + interpret in one). Wraps GatedPatchwork.

    Reads:  cache['tri']       — (B, A)
            cache['embedding'] — (B, D) optional
            cache['anchors_n'] — (A, D) optional
    Writes: cache['patchwork']   — (B, pw_dim)
            cache['gate_info']   — dict diagnostics
    """

    def __init__(self, name, n_anchors, n_comp=8, d_comp=64, dim=256,
                 strategy='cm_gate', n_neighbors=3,
                 activation='squared_relu', **kwargs):
        super().__init__(name, **kwargs)
        self.inner = GatedPatchwork(
            n_anchors, n_comp, d_comp, dim,
            strategy, n_neighbors, activation)

    def forward(self, tri=None, embedding=None, anchors=None):
        if self.parent is not None:
            if tri is None:
                tri = self.parent.cache_get('tri')
            if embedding is None:
                embedding = self.parent.cache_get('embedding')
            if anchors is None:
                anchors = self.parent.cache_get('anchors_n')

        pw, gate_info = self.inner(tri, embedding, anchors)

        if self.parent is not None:
            self.parent.cache_set('patchwork', pw)
            self.parent.cache_set('gate_info', gate_info)
        return pw

    @property
    def output_dim(self):
        return self.inner.output_dim
