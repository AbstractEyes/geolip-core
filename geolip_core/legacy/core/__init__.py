"""
core — GeoLIP geometric observer framework.

Six-stage paradigm: Input, Mutation, Association, Curation, Distinction, Loss.
GeoLIP composes stages. Constellation is one concrete set of implementations.
"""

# Stage interfaces + GeoLIP loop
from .observer import Input, Mutation, Association, Curation, Distinction, GeoLIP

# Constellation implementations
from .constellation import (
    Constellation, ConstellationAssociation, ConstellationCuration,
    ConstellationObserver,
    init_anchors_xavier, init_anchors_orthogonal,
    init_anchors_repulsion, INIT_METHODS,
)

# Activation
from .activation import SquaredReLU, StarReLU, make_activation, ACTIVATIONS

# Patchwork + Magnitude + Push
from .patchwork import Patchwork, MagnitudeFlow, AnchorPush

# Relay
from .constellation_relay import RelayLayer, ConstellationRelay

# Route
from .constellation_route import FlowAttention

# Core utilities
from .core import GeometricAutograd, param_count, model_summary

# Memory
from .memory import EmbeddingBuffer

# Losses
from .losses import (
    cv_loss, cv_metric, cv_multi_scale, cayley_menger_vol2,
    nce_loss, ce_loss, ce_loss_paired,
    bridge_loss, bridge_loss_paired,
    assign_bce_loss, assign_nce_loss,
    attraction_loss, spread_loss, knn_accuracy,
    three_domain_loss, observer_loss,
)

__all__ = [
    # Stage interfaces
    'Input', 'Mutation', 'Association', 'Curation', 'Distinction',
    # Composition
    'GeoLIP',
    # Constellation
    'Constellation', 'ConstellationAssociation', 'ConstellationCuration',
    'ConstellationObserver',
    'init_anchors_xavier', 'init_anchors_orthogonal',
    'init_anchors_repulsion', 'INIT_METHODS',
    # Activation
    'SquaredReLU', 'StarReLU', 'make_activation', 'ACTIVATIONS',
    # Patchwork + Magnitude + Push
    'Patchwork', 'MagnitudeFlow', 'AnchorPush',
    # Relay
    'RelayLayer', 'ConstellationRelay',
    # Route
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
    'three_domain_loss', 'observer_loss',
]