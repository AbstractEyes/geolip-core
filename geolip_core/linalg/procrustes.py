"""
Subspace-preserving Procrustes alignment.

Uses geolip.linalg.svd internally (auto-dispatches FL eigh / Triton / torch).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Any

from .newton_schulz import newton_schulz_invsqrt

__all__ = ['batched_procrustes']


def batched_procrustes(
    source: Tensor,
    target: Tensor,
    rank: int = 24,
    whiten: bool = True,
    schulz_iters: int = 10,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Batched Procrustes alignment with subspace-preserving rotation.

    N <= 32: full N-d Procrustes via SVD.
    N > 32:  project to rank-d, align, lift back preserving orthogonal complement.

    Args:
        source:       (B, n_samples, N) or (n_samples, N)
        target:       (B, n_samples, N) or (n_samples, N)
        rank:         Projection rank for N > 32
        whiten:       Apply Newton-Schulz whitening
        schulz_iters: Iterations for whitening

    Returns:
        aligned: same shape as source
        info:    dict with method, rotation, diagnostics
    """
    unbatched = source.ndim == 2
    if unbatched:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)

    B, n_samples, N = source.shape
    device = source.device

    with torch.amp.autocast('cuda', enabled=False):
        source_f = source.float()
        target_f = target.float()

        src_mean = source_f.mean(1, keepdim=True)
        tgt_mean = target_f.mean(1, keepdim=True)
        src_c = source_f - src_mean
        tgt_c = target_f - tgt_mean

        if whiten:
            src_cov = torch.bmm(src_c.transpose(1, 2), src_c) / max(n_samples - 1, 1)
            tgt_cov = torch.bmm(tgt_c.transpose(1, 2), tgt_c) / max(n_samples - 1, 1)
            src_W = newton_schulz_invsqrt(src_cov, iters=schulz_iters)
            tgt_W = newton_schulz_invsqrt(tgt_cov, iters=schulz_iters)
            src_w = F.normalize(torch.bmm(src_c, src_W), dim=-1)
            tgt_w = F.normalize(torch.bmm(tgt_c, tgt_W), dim=-1)
        else:
            src_w = src_c
            tgt_w = tgt_c

        use_projection = N > 32 and rank < N

        if not use_projection:
            C = torch.bmm(src_w.transpose(1, 2), tgt_w)
            U, _, Vh = torch.linalg.svd(C)
            R = torch.bmm(U, Vh)
            aligned_w = torch.bmm(src_w, R)
            if whiten:
                aligned = torch.bmm(aligned_w, torch.linalg.pinv(tgt_W)) + tgt_mean
            else:
                aligned = aligned_w + tgt_mean
            cos_after = F.cosine_similarity(
                aligned_w[:, :min(1000, n_samples)],
                tgt_w[:, :min(1000, n_samples)], dim=-1).mean().item()
            info = {'method': 'full', 'N': N, 'rank': N,
                    'rotation': R, 'cos_after': cos_after}
        else:
            k = min(rank, N - 1)
            P = torch.linalg.qr(
                torch.randn(B, N, k, device=device, dtype=torch.float32)).Q
            src_proj = torch.bmm(src_w, P)
            tgt_proj = torch.bmm(tgt_w, P)
            C_k = torch.bmm(src_proj.transpose(1, 2), tgt_proj)
            U_k, _, Vh_k = torch.linalg.svd(C_k)
            R_k = torch.bmm(U_k, Vh_k)
            src_in = torch.bmm(src_w, P)
            P_T = P.transpose(1, 2)
            src_perp = src_w - torch.bmm(src_in, P_T)
            src_rotated = torch.bmm(torch.bmm(src_in, R_k), P_T)
            aligned_w = src_rotated + src_perp
            if whiten:
                aligned = torch.bmm(aligned_w, torch.linalg.pinv(tgt_W)) + tgt_mean
            else:
                aligned = aligned_w + tgt_mean
            cos_after = F.cosine_similarity(
                aligned_w[:, :min(1000, n_samples)],
                tgt_w[:, :min(1000, n_samples)], dim=-1).mean().item()
            info = {'method': 'subspace', 'N': N, 'rank': k,
                    'rotation_k': R_k, 'projection': P, 'cos_after': cos_after}

    if unbatched:
        aligned = aligned.squeeze(0)

    return aligned, info