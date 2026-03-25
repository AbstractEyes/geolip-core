"""
Shared utilities for the geometric pipeline.

Activations, parameter counting, model summary, GeometricAutograd,
and empirical constants.

Usage:
    from geolip_core.core.util import make_activation, param_count, GeometricAutograd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ACTIVATIONS ──────────────────────────────────────────────────────────────

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
    """Factory for activation functions."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]()


# ── GEOMETRIC AUTOGRAD ───────────────────────────────────────────────────────

class GeometricAutograd(torch.autograd.Function):
    """Manifold-aware gradient correction on S^(D-1).

    Forward: identity (pass-through).
    Backward: project gradients to tangent plane + optional separation penalty.
    """

    @staticmethod
    def forward(ctx, emb, anchors, tang_strength, sep_strength):
        ctx.save_for_backward(emb, anchors)
        ctx.tang = tang_strength
        ctx.sep = sep_strength
        return emb

    @staticmethod
    def backward(ctx, grad):
        emb, anchors = ctx.saved_tensors
        dot = (grad * emb).sum(-1, keepdim=True)
        corrected = grad - ctx.tang * dot * emb
        if ctx.sep > 0:
            an = F.normalize(anchors.detach(), dim=-1)
            nearest = an[(emb @ an.T).argmax(-1)]
            toward = (corrected * nearest).sum(-1, keepdim=True)
            corrected = corrected - ctx.sep * F.relu(toward) * nearest
        return corrected, None, None, None


# ── PARAMETER COUNTING ───────────────────────────────────────────────────────

def param_count(module, name=""):
    """Count total and trainable parameters."""
    t = sum(p.numel() for p in module.parameters())
    tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if name:
        print(f"  {name}: {t:,} ({tr:,} trainable)")
    return t, tr


def model_summary(model):
    """Print parameter breakdown by top-level children."""
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total:,}")
    for n, m in model.named_children():
        c = sum(p.numel() for p in m.parameters())
        if c > 0:
            print(f"    {n}: {c:,}")
    return total


# ── EMPIRICAL CONSTANTS ──────────────────────────────────────────────────────

CV_PENTACHORON_BAND = (0.20, 0.23)       # Universal across 17+ architectures
BINDING_BOUNDARY = 0.29154                 # Binding/separation phase boundary
SEPARATION_COMPLEMENT = 0.70846            # 1 - BINDING_BOUNDARY
EFFECTIVE_GEO_DIM = 16                     # S^15
IRREDUCIBLE_CV_MIN = 0.125                 # Minimum CV on sphere