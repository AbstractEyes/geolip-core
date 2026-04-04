"""
pipeline/components — TorchComponent wrappers for every core module.

Each wrapper is a thin cache adapter. Core modules do the geometry.
Wrappers handle the bus: read named cache slots, delegate to core,
write results back. The pipeline orchestrates execution order.

Usage:
    from geolip_core.pipeline.components import ObserveSVD, CurateCMGate
"""

from .observe_svd import ObserveSVD, ObserveSVDTokens
from .associate_constellation import AssociateConstellation
from .mutate_relay import MutateRelay
from .mutate_flow import MutateFlow
from .curate_gate import CurateCMGate
from .curate_patchwork import CuratePatchwork, CurateGatedPatchwork
from .curate_magnitude import CurateMagnitude
from .align_procrustes import AlignProcrustes
from .fuse import FuseGeometric

# Geometric Transformer components (extracted from geometric_transformer.py)
from .curate_cm_validated import CMValidatedGate
from .distinguish_nce_bank import GeoResidualBank
from .context_film import FiLMLayer
from .align_cayley import CayleyOrthogonal
from .compose_quaternion import QuaternionCompose
from .project_manifold import ManifoldProjection
from .context_position_geometric import PositionGeometricContext
from .attend_geometric import GeometricAttention
from .attend_content import ContentAttention
from .layer_geometric import GeometricTransformerLayer
from .transformer_geometric import (
    GeometricTransformer,
    geo_transformer_esm2,
    geo_transformer_small,
    geo_transformer_vision,
)
