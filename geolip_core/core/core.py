"""
Core utilities — GeometricAutograd, parameter counting, model summary.

Usage:
    from geolip_core.core.core import GeometricAutograd, param_count, model_summary
"""

import torch
import torch.nn.functional as F


class GeometricAutograd(torch.autograd.Function):
    """Manifold-aware gradient correction on S^(D-1).

    Forward: identity (pass-through).
    Backward: project gradients to tangent plane + optional separation penalty.

    The tangent projection removes the radial component of the gradient,
    keeping updates on the manifold. The separation penalty pushes gradients
    away from the nearest anchor, preventing collapse.

    Usage:
        emb = GeometricAutograd.apply(emb, constellation.anchors, 1.0, 0.1)
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


def param_count(module, name=""):
    """Count total and trainable parameters.

    Args:
        module: nn.Module
        name: optional label for printing

    Returns:
        (total, trainable) tuple
    """
    t = sum(p.numel() for p in module.parameters())
    tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if name:
        print(f"  {name}: {t:,} ({tr:,} trainable)")
    return t, tr


def model_summary(model):
    """Print parameter breakdown by top-level children.

    Returns:
        int: total parameter count
    """
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total:,}")
    for n, m in model.named_children():
        c = sum(p.numel() for p in m.parameters())
        if c > 0:
            print(f"    {n}: {c:,}")
    return total