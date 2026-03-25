"""
curate — Select what matters from associations.

CM gate provides geometric attention (simplex volume as relevance).
Patchwork interprets curated triangulation through compartments.
MagnitudeFlow weights associations by importance.
"""

from .gate import AnchorGate, GatedPatchwork, cayley_menger_det, cm_validity_score
from .patchwork import Patchwork, MagnitudeFlow, AnchorPush
