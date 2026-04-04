"""
FiLMLayer — Feature-wise Linear Modulation.
Proven in Ryan Spearman (rho=0.309, 76/84 wins).
"""

import torch.nn as nn
from geolip_core.pipeline.observer import TorchComponent


class FiLMLayer(TorchComponent):
    """Feature-wise Linear Modulation. Near-identity-initialized.
    gamma ≈ 1 + 0.01·geo_ctx, beta ≈ 0.01·geo_ctx at init.
    Gradient flows through to geo_ctx from step 0.
    """
    def __init__(self, name, feature_dim, context_dim):
        super().__init__(name)
        self.to_gamma = nn.Linear(context_dim, feature_dim)
        self.to_beta = nn.Linear(context_dim, feature_dim)
        nn.init.normal_(self.to_gamma.weight, std=0.01); nn.init.ones_(self.to_gamma.bias)
        nn.init.normal_(self.to_beta.weight, std=0.01); nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, ctx):
        return self.to_gamma(ctx) * x + self.to_beta(ctx)
