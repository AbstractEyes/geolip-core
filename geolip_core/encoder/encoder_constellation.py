"""
Conv Encoder — 8-layer conv backbone + full GeoLIP image classification pipeline.

ConvEncoder:        Simple conv backbone. Feature extraction only.
GeoLIPConvEncoder:  ConvEncoder → S^(d-1) → MagnitudeFlow → ConstellationCore → task head.

The task head and CE loss live HERE, not in the observer.
ConstellationCore observes. This class classifies.

Usage:
    from encoder.encoder_conv import ConvEncoder, GeoLIPConvEncoder
"""