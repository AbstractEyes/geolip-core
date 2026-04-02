"""
geolip_core — Geometric observer framework for deep learning.

Structure:
    core/       Geometric behaviors (input, associate, curate, align, distinguish)
    pipeline/   Composed geometric substrates (layer, backbone, GeoLIP)
    example/    Working models built with the pipeline
    analysis/   Diagnostic tools
    utils/      Engineering infrastructure (kernels, linalg, memory)

Quick start:
    from geolip_core.core import Constellation, Patchwork, SVDObserver
    from geolip_core.pipeline import Input, GeoLIP
    from geolip_core.utils import gram_eigh_svd, batched_procrustes
"""

__version__ = "0.2.0"
import sys
sys.modules['geolip'] = sys.modules[__name__]