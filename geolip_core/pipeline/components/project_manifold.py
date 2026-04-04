"""
ManifoldProjection — project transformer hidden states to S^(d-1).
"""

import torch.nn as nn
import torch.nn.functional as F
from geolip_core.pipeline.observer import TorchComponent


class ManifoldProjection(TorchComponent):
    """Input stage: project transformer hidden states to S^(d-1)."""
    def __init__(self, name, d_model, manifold_dim):
        super().__init__(name)
        self.proj = nn.Linear(d_model, manifold_dim)
        self.norm = nn.LayerNorm(manifold_dim)

    def forward(self, hidden_states):
        h = self.norm(self.proj(hidden_states))
        return F.normalize(h, dim=-1)
