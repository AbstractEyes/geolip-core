"""
encoder — Image encoder variants for the GeoLIP pipeline.
"""

from .encoder_conv import ConvEncoder, GeoLIPConvEncoder

__all__ = [
    'ConvEncoder',
    'GeoLIPConvEncoder',
]