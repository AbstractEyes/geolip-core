"""
encoder — Input stages and full-stack compositions for the GeoLIP pipeline.

Partial stacks (Input stages):
    ConvEncoder         — pixels → S^(d-1)
    (wavelet, scatter, transformer — future)

Full stack:
    ConstellationEncoder — GeoLIP with constellation stages, modality-agnostic
    ClassificationHead   — Distinction stage for supervised classification
    GeoLIPEncoder        — Any Input + ConstellationEncoder composed

Legacy:
    GeoLIPConvEncoder    — ConvEncoder + inline pipeline (pre-paradigm)
"""

from .encoder_conv import ConvEncoder, GeoLIPConvEncoder
from .encoder_constellation import ConstellationEncoder, ClassificationHead, GeoLIPEncoder

__all__ = [
    # Input stages
    'ConvEncoder',
    # Full stack (paradigm)
    'ConstellationEncoder',
    'ClassificationHead',
    'GeoLIPEncoder',
    # Legacy
    'GeoLIPConvEncoder',
]