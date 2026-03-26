"""
FuseGeometric — Pipeline-specific fusion TorchComponent.

No core equivalent — fusion is pipeline-level composition.
Combines all geometric signals into a unified feature vector.

Reads: svd_features, patchwork, embedding
Writes: svd_context, geo_features
"""

import torch
import torch.nn as nn

from geofractal.router.components.torch_component import TorchComponent


class FuseGeometric(TorchComponent):
    """Fuse all geometric signals into unified feature vector.

    Reads:  cache['svd_features'] — (B, svd_feat_dim)
            cache['patchwork']    — (B, pw_dim)
            cache['embedding']    — (B, embed_dim)
    Writes: cache['svd_context']  — (B, pw_dim) projected SVD
            cache['geo_features'] — (B, feature_dim) unified output
    """

    def __init__(self, name, svd_feat_dim, pw_dim, embed_dim, **kwargs):
        super().__init__(name, **kwargs)
        self.svd_proj = nn.Sequential(
            nn.Linear(svd_feat_dim, pw_dim),
            nn.LayerNorm(pw_dim), nn.GELU())
        self._feature_dim = pw_dim + pw_dim + embed_dim

    def forward(self, svd_features=None, patchwork=None, embedding=None):
        if self.parent is not None:
            if svd_features is None:
                svd_features = self.parent.cache_get('svd_features')
            if patchwork is None:
                patchwork = self.parent.cache_get('patchwork')
            if embedding is None:
                embedding = self.parent.cache_get('embedding')

        svd_context = self.svd_proj(svd_features)
        features = torch.cat([svd_context, patchwork, embedding], dim=-1)

        if self.parent is not None:
            self.parent.cache_set('svd_context', svd_context)
            self.parent.cache_set('geo_features', features)
        return features

    @property
    def feature_dim(self):
        return self._feature_dim
