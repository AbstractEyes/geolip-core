"""
ObserveSVD / ObserveSVDTokens — Input stage TorchComponents.

Thin cache adapters around core.input.svd.SVDObserver and SVDTokenObserver.

Writes: svd_S, svd_Vh, svd_features, svd_novelty
"""

from geofractal.router.components.torch_component import TorchComponent
from geolip_core.core.input.svd import SVDObserver, SVDTokenObserver


class ObserveSVD(TorchComponent):
    """Observe spatial features via SVD. Wraps SVDObserver.

    For Conv backbones: (B, C, H, W) → structural decomposition.

    Writes: cache['svd_S']        — (B, k) singular values
            cache['svd_Vh']       — (B, k, k) rotation matrix
            cache['svd_features'] — (B, 2k+2) compact summary
            cache['svd_novelty']  — (B, k) EMA deviation
    """

    def __init__(self, name, in_channels, svd_rank=24, **kwargs):
        super().__init__(name, **kwargs)
        self.inner = SVDObserver(in_channels, svd_rank)

    def forward(self, x):
        S, Vh, features, novelty = self.inner(x)
        if self.parent is not None:
            self.parent.cache_set('svd_S', S)
            self.parent.cache_set('svd_Vh', Vh)
            self.parent.cache_set('svd_features', features)
            self.parent.cache_set('svd_novelty', novelty)
        return features

    @property
    def feature_dim(self):
        return self.inner.feature_dim

    def update_ema(self, S, Vh):
        self.inner.update_ema(S, Vh)


class ObserveSVDTokens(TorchComponent):
    """Observe token sequence structure via SVD. Wraps SVDTokenObserver.

    For sequence inputs: (B, seq, dim) → structural decomposition.

    Writes: cache['svd_S']        — (B, seq) singular values
            cache['svd_Vh']       — (B, seq, seq) rotation matrix
            cache['svd_features'] — (B, 2*seq+2) compact summary
            cache['svd_novelty']  — (B, seq) EMA deviation
    """

    def __init__(self, name, seq_len, **kwargs):
        super().__init__(name, **kwargs)
        self.inner = SVDTokenObserver(seq_len)

    def forward(self, x):
        S, Vh, features, novelty = self.inner(x)
        if self.parent is not None:
            self.parent.cache_set('svd_S', S)
            self.parent.cache_set('svd_Vh', Vh)
            self.parent.cache_set('svd_features', features)
            self.parent.cache_set('svd_novelty', novelty)
        return features

    @property
    def feature_dim(self):
        return self.inner.feature_dim

    def update_ema(self, S, Vh):
        self.inner.update_ema(S, Vh)
