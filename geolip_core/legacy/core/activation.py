"""
Activation functions for geometric pipelines.

Usage:
    from geolip_core.core.activation import make_activation, SquaredReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    """ReLU² — simple, effective for geometric pipelines."""
    def forward(self, x):
        return F.relu(x) ** 2


class StarReLU(nn.Module):
    """Learned scale/bias variant of ReLU²."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 0.8944)
        self.bias = nn.Parameter(torch.zeros(1) - 0.4472)

    def forward(self, x):
        return F.relu(x) ** 2 * self.scale + self.bias


ACTIVATIONS = {
    'squared_relu': SquaredReLU,
    'star_relu': StarReLU,
    'gelu': lambda: nn.GELU(),
    'relu': lambda: nn.ReLU(),
    'sigmoid': lambda: nn.Sigmoid(),
}


def make_activation(name='squared_relu'):
    """Factory for activation functions.

    Args:
        name: one of 'squared_relu', 'star_relu', 'gelu', 'relu', 'sigmoid'

    Returns:
        nn.Module instance
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()