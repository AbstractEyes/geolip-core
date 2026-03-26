"""
MutateRelay — Mutation stage TorchComponent.

Thin cache adapter around core.associate.relay.ConstellationRelay.

Writes: relay_output, relay_tri
"""

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.associate.relay import ConstellationRelay


class MutateRelay(TorchComponent):
    """Per-token geometric mutation via relay stack. Wraps ConstellationRelay.

    O(S) complexity. Handles (B, D) or (B, S, D).

    Writes: cache['relay_output'] — same shape as input
            cache['relay_tri']    — (B, [S,] A) triangulation (if return_tri)
    """

    def __init__(self, name, dim, n_anchors=16, n_comp=8, d_comp=64,
                 gate_init=-3.0, anchor_init='repulsion',
                 activation='squared_relu', **kwargs):
        super().__init__(name, **kwargs)
        self.inner = ConstellationRelay(
            dim, n_anchors, n_comp, d_comp,
            gate_init, anchor_init, activation)

    def forward(self, x, return_tri=False):
        result = self.inner(x, return_tri=return_tri)
        if return_tri:
            out, tri = result
            if self.parent is not None:
                self.parent.cache_set('relay_output', out)
                self.parent.cache_set('relay_tri', tri)
            return out, tri
        else:
            if self.parent is not None:
                self.parent.cache_set('relay_output', result)
            return result
