"""
core — Geometric building blocks for the GeoLIP ecosystem.

Constellation triangulation, patchwork compartments, relay layers,
losses, metrics, and anchor management.
"""

from .activation import SquaredReLU, StarReLU, make_activation, ACTIVATIONS
from .constellation import (
    Constellation, init_anchors_xavier, init_anchors_orthogonal,
    init_anchors_repulsion, INIT_METHODS,
)
from .patchwork import Patchwork, MagnitudeFlow, AnchorPush
from .constellation_relay import RelayLayer, ConstellationRelay
from .constellation_route import FlowAttention
from .core import GeometricAutograd, param_count, model_summary
from .memory import EmbeddingBuffer
from .losses import (
    cv_loss, cv_metric, cv_multi_scale, cayley_menger_vol2,
    nce_loss, ce_loss, ce_loss_paired,
    bridge_loss, bridge_loss_paired,
    assign_bce_loss, assign_nce_loss,
    attraction_loss, spread_loss, knn_accuracy,
    three_domain_loss,
)

__all__ = [
    # Activation
    'SquaredReLU', 'StarReLU', 'make_activation', 'ACTIVATIONS',
    # Constellation
    'Constellation', 'init_anchors_xavier', 'init_anchors_orthogonal',
    'init_anchors_repulsion', 'INIT_METHODS',
    # Patchwork + Magnitude + Push
    'Patchwork', 'MagnitudeFlow', 'AnchorPush',
    # Relay
    'RelayLayer', 'ConstellationRelay',
    # Route (historical)
    'FlowAttention',
    # Core utilities
    'GeometricAutograd', 'param_count', 'model_summary',
    # Memory
    'EmbeddingBuffer',
    # Losses
    'cv_loss', 'cv_metric', 'cv_multi_scale', 'cayley_menger_vol2',
    'nce_loss', 'ce_loss', 'ce_loss_paired',
    'bridge_loss', 'bridge_loss_paired',
    'assign_bce_loss', 'assign_nce_loss',
    'attraction_loss', 'spread_loss', 'knn_accuracy',
    'three_domain_loss',
]