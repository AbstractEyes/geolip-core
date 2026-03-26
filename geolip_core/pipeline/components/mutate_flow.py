"""
MutateFlow — Mutation stage TorchComponent.

Thin cache adapter around core.associate.route.FlowAttention.

Writes: flow_output
"""

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.associate.route import FlowAttention


class MutateFlow(TorchComponent):
    """ODE flow mutation in tangent space. Wraps FlowAttention.

    Reads:  cache['embedding'] or direct input
    Writes: cache['flow_output'] — (B, D) corrected embedding on S^(d-1)
    """

    def __init__(self, name, dim, n_anchors, flow_dim=64,
                 n_steps=3, gate_init=-3.0, **kwargs):
        super().__init__(name, **kwargs)
        self.inner = FlowAttention(
            dim, n_anchors, flow_dim, n_steps, gate_init=gate_init)

    def forward(self, emb=None, constellation=None):
        if emb is None and self.parent is not None:
            emb = self.parent.cache_get('embedding')
        result = self.inner(emb, constellation)
        if self.parent is not None:
            self.parent.cache_set('flow_output', result)
        return result
