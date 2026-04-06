"""
core — GeoLIP geometric behavior package.

Five stages, five directories:
    input/       Data-type ingestion and observation
    associate/   Measure relationships to reference frame
    curate/      Select what matters from associations
    align/       How spaces relate to each other
    distinguish/ Task-specific output and training signal

Usage:
    from geolip_core.core import Constellation, Patchwork, AnchorGate
    from geolip_core.core import cv_loss, ce_loss, knn_accuracy
    from geolip_core.core import compute_target_cv, soft_hand_loss
    from geolip_core.core import SVDObserver, ProcrustesAlignment
    from geolip_core.core import make_activation, param_count
"""

# ── Shared utilities ──
from .util import (
    SquaredReLU, StarReLU, make_activation, ACTIVATIONS,
    GeometricAutograd, param_count, model_summary,
    CV_PENTACHORON_BAND, BINDING_BOUNDARY, SEPARATION_COMPLEMENT,
    EFFECTIVE_GEO_DIM, IRREDUCIBLE_CV_MIN,
)

# ── Input: data-type ingestion ──
from .input.svd import SVDObserver, SVDTokenObserver

# ── Associate: measure relationships ──
from .associate import (
    Constellation, ConstellationAssociation, ConstellationCuration,
    ConstellationObserver,
    init_anchors_xavier, init_anchors_orthogonal,
    init_anchors_repulsion, INIT_METHODS,
    RelayLayer, ConstellationRelay,
    FlowAttention,
)

# ── Curate: select what matters ──
from .curate import (
    AnchorGate, GatedPatchwork, cayley_menger_det, cm_validity_score,
    Patchwork, MagnitudeFlow, AnchorPush,
)

# ── Align: how spaces relate ──
from .align import ProcrustesAlignment

# ── Distinguish: task output + losses ──
from .distinguish import (
    cv_loss, cv_metric, cv_multi_scale, cayley_menger_vol2,
    nce_loss, ce_loss, ce_loss_paired,
    bridge_loss, bridge_loss_paired,
    assign_bce_loss, assign_nce_loss,
    attraction_loss, spread_loss, knn_accuracy,
    three_domain_loss, observer_loss,
    compute_target_cv, compute_target_cv_with_stats,
    cv_proximity, soft_hand_weights, soft_hand_loss,
)