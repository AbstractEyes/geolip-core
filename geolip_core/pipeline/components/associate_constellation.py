"""
AssociateConstellation — Associate stage TorchComponent.

Thin cache adapter around core.associate.constellation.ConstellationAssociation.

Writes: embedding, anchors_n, cos, tri, nearest, assignment
"""

import torch.nn.functional as F

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.associate.constellation import ConstellationAssociation


class AssociateConstellation(TorchComponent):
    """Triangulate against learned anchors. Wraps ConstellationAssociation.

    Reads:  (direct input: embedding on S^(d-1))
    Writes: cache['embedding']       — (B, dim) L2-normalized
            cache['anchors_n']       — (A, dim) normalized anchors
            cache['cos']             — (B, A) cosine similarities
            cache['tri']             — (B, A) triangulation distances
            cache['nearest']         — (B,) closest anchor index
            cache['assignment']      — (B, A) soft assignment
    """

    def __init__(self, name, dim=256, n_anchors=128, anchor_drop=0.15,
                 anchor_init='repulsion', assign_temp=0.1, **kwargs):
        super().__init__(name, **kwargs)
        self.inner = ConstellationAssociation(
            dim=dim, n_anchors=n_anchors, anchor_drop=anchor_drop,
            anchor_init=anchor_init, assign_temp=assign_temp)

    def forward(self, emb, **context):
        a_out = self.inner.associate(emb, **context)
        if self.parent is not None:
            self.parent.cache_set('embedding', emb)
            self.parent.cache_set('anchors_n',
                F.normalize(self.inner.constellation.anchors, dim=-1))
            self.parent.cache_set('cos', a_out['cos_to_anchors'])
            self.parent.cache_set('tri', a_out['distances'])
            self.parent.cache_set('nearest', a_out['nearest'])
            self.parent.cache_set('assignment', a_out['assignment'])
        return a_out

    @property
    def constellation(self):
        return self.inner.constellation

    @property
    def anchors(self):
        return self.inner.constellation.anchors
