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
