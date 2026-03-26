"""
CurateCMGate — Curate stage TorchComponent.

Thin cache adapter around core.curate.gate.AnchorGate.

Reads: embedding, anchors_n, tri
Writes: gate_values, gate_info, tri_gated
"""

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.curate.gate import AnchorGate


class CurateCMGate(TorchComponent):
    """CM validity anchor selection. Wraps AnchorGate.

    Reads:  cache['embedding']  — (B, dim)
            cache['anchors_n']  — (A, dim)
            cache['tri']        — (B, A)
    Writes: cache['gate_values'] — (B, A) per-anchor gate [0,1]
            cache['gate_info']   — dict diagnostics
            cache['tri_gated']   — (B, A) gated triangulation
    """

    def __init__(self, name, n_anchors, embed_dim, n_comp=8,
                 n_neighbors=3, strategy='cm_gate',
                 top_k=None, top_p=0.9, **kwargs):
        super().__init__(name, **kwargs)
        self.inner = AnchorGate(
            n_anchors, embed_dim, n_comp, n_neighbors,
            strategy, top_k, top_p)

    def forward(self, embedding=None, anchors=None, tri=None):
        if self.parent is not None:
            if embedding is None:
                embedding = self.parent.cache_get('embedding')
            if anchors is None:
                anchors = self.parent.cache_get('anchors_n')
            if tri is None:
                tri = self.parent.cache_get('tri')

        gate_values, gate_assign, gate_info = self.inner(embedding, anchors, tri)
        tri_gated = tri * gate_values

        if self.parent is not None:
            self.parent.cache_set('gate_values', gate_values)
            self.parent.cache_set('gate_info', gate_info)
            self.parent.cache_set('tri_gated', tri_gated)

        return tri_gated
