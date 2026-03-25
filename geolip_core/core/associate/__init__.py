"""
associate — Measure relationships to reference frame.

Constellation anchors triangulate embeddings on S^(d-1).
Relay and route provide manifold-aware mutation through association.
"""

from .constellation import (
    Constellation, ConstellationAssociation, ConstellationCuration,
    ConstellationObserver,
    init_anchors_xavier, init_anchors_orthogonal,
    init_anchors_repulsion, INIT_METHODS,
)
from .relay import RelayLayer, ConstellationRelay
from .route import FlowAttention
