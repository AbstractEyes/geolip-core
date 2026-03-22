"""
GeoLIP Losses & Regularization
=================================
Every loss and metric in the GeoLIP pipeline, with uniform interfaces.

All loss functions: (inputs) → scalar tensor (differentiable)
All metrics: (inputs) → float (non-differentiable, for monitoring)

CV functions default to batched computation (141x speedup).
Set batched=False for sequential fallback.

Loss Spectrum (3 domains):
  EXTERNAL:   ce_loss, nce_loss (embedding-level)
  GEOMETRIC:  nce_loss (patchwork), bridge_loss
  INTERNAL:   assign_bce, assign_nce, nce_loss (triangulation),
              attraction_loss, cv_loss, spread_loss

Metrics:
  cv_metric, cv_multi_scale, cayley_menger_vol2

Compound:
  three_domain_loss — the full cooperative loss from InternalConstellationCore

Usage:
    from geolip_losses import cv_loss, cv_metric, nce_loss, three_domain_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════
# CV — Coefficient of Variation of Pentachoron Volumes
# ══════════════════════════════════════════════════════════════════

def _batch_pentachoron_volumes(emb, n_samples=200, n_points=5):
    """Compute pentachoron volumes in one batched operation. Zero Python loops.

    Args:
        emb: (N, D) embeddings on S^(d-1)
        n_samples: random pentachora to sample
        n_points: points per simplex (5 = pentachoron)

    Returns:
        (n_valid,) tensor of simplex volumes
    """
    N, D = emb.shape
    device, dtype = emb.device, emb.dtype
    pool = min(N, 512)

    # Batched randperm via argsort on random values
    indices = torch.rand(n_samples, pool, device=device).argsort(dim=1)[:, :n_points]
    pts = emb[:pool][indices]  # (n_samples, n_points, D)

    gram = torch.bmm(pts, pts.transpose(1, 2))
    norms = torch.diagonal(gram, dim1=1, dim2=2)
    d2 = F.relu(norms.unsqueeze(2) + norms.unsqueeze(1) - 2 * gram)

    M = n_points + 1
    cm = torch.zeros(n_samples, M, M, device=device, dtype=dtype)
    cm[:, 0, 1:] = 1.0
    cm[:, 1:, 0] = 1.0
    cm[:, 1:, 1:] = d2

    k = n_points - 1
    pf = ((-1.0) ** (k + 1)) / ((2.0 ** k) * (math.factorial(k) ** 2))
    dets = pf * torch.linalg.det(cm.float())

    valid = dets > 1e-20
    return dets[valid].to(dtype).sqrt()


def _sequential_pentachoron_volumes(emb, n_samples=200, n_points=5):
    """Sequential fallback. One det call per sample."""
    N = emb.shape[0]
    device, dtype = emb.device, emb.dtype
    vols = []
    for _ in range(n_samples):
        idx = torch.randperm(min(N, 512), device=device)[:n_points]
        pts = emb[idx].unsqueeze(0)
        gram = torch.bmm(pts, pts.transpose(1, 2))
        norms = torch.diagonal(gram, dim1=1, dim2=2)
        d2 = F.relu(norms.unsqueeze(2) + norms.unsqueeze(1) - 2 * gram)
        M = n_points + 1
        cm = torch.zeros(1, M, M, device=device, dtype=dtype)
        cm[:, 0, 1:] = 1; cm[:, 1:, 0] = 1; cm[:, 1:, 1:] = d2
        k = n_points - 1
        pf = ((-1.0) ** (k + 1)) / ((2.0 ** k) * (math.factorial(k) ** 2))
        v2 = pf * torch.linalg.det(cm.float())
        if v2[0].item() > 1e-20:
            vols.append(v2[0].to(dtype).sqrt())
    if len(vols) < 5:
        return torch.tensor([], device=device, dtype=dtype)
    return torch.stack(vols)


def cv_loss(emb, target=0.22, n_samples=64, n_points=5, batched=True):
    """Differentiable CV loss. Returns (CV - target)².

    Args:
        emb: (N, D) L2-normalized embeddings
        target: CV target (0.22 = natural basin of S^(d-1) at eff_dim ~16)
        n_samples: pentachora to sample (32-64 for training)
        n_points: points per simplex
        batched: use batched computation (141x faster, default True)

    Returns:
        scalar tensor, differentiable w.r.t. emb
    """
    if emb.shape[0] < n_points:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)

    if batched:
        vols = _batch_pentachoron_volumes(emb, n_samples, n_points)
    else:
        vols = _sequential_pentachoron_volumes(emb, n_samples, n_points)

    if vols.shape[0] < 5:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    cv = vols.std() / (vols.mean() + 1e-8)
    return (cv - target).pow(2)


def cv_metric(emb, n_samples=200, n_points=5, batched=True):
    """Non-differentiable CV for monitoring. Target band: 0.20–0.23.

    Returns:
        float: coefficient of variation of simplex volumes
    """
    with torch.no_grad():
        if batched:
            vols = _batch_pentachoron_volumes(emb, n_samples, n_points)
        else:
            vols = _sequential_pentachoron_volumes(emb, n_samples, n_points)
        if vols.shape[0] < 10:
            return 0.0
        return (vols.std() / (vols.mean() + 1e-8)).item()


def cv_multi_scale(emb, scales=(3, 4, 5, 6, 7, 8), n_samples=100, batched=True):
    """CV at multiple simplex sizes. Returns dict: {n_points: cv_value}.

    Healthy geometry: all scales in [0.18, 0.25].
    """
    results = {}
    with torch.no_grad():
        for n_pts in scales:
            if batched:
                vols = _batch_pentachoron_volumes(emb, n_samples, n_pts)
            else:
                vols = _sequential_pentachoron_volumes(emb, n_samples, n_pts)
            if vols.shape[0] >= 10:
                results[n_pts] = round((vols.std() / (vols.mean() + 1e-8)).item(), 4)
            else:
                results[n_pts] = None
    return results


def cayley_menger_vol2(points):
    """Squared simplex volume. points: (B, N, D) → (B,)."""
    B, N, D = points.shape
    gram = torch.bmm(points, points.transpose(1, 2))
    norms = torch.diagonal(gram, dim1=1, dim2=2)
    d2 = F.relu(norms.unsqueeze(2) + norms.unsqueeze(1) - 2 * gram)
    cm = torch.zeros(B, N + 1, N + 1, device=points.device, dtype=points.dtype)
    cm[:, 0, 1:] = 1; cm[:, 1:, 0] = 1; cm[:, 1:, 1:] = d2
    k = N - 1
    sign = (-1.0) ** (k + 1)
    fact = math.factorial(k)
    return sign * torch.linalg.det(cm.float()).to(points.dtype) / ((2 ** k) * (fact ** 2))


# ══════════════════════════════════════════════════════════════════
# NCE — InfoNCE contrastive loss
# ══════════════════════════════════════════════════════════════════

def nce_loss(z1, z2, temperature=0.07, normalize=True):
    """Symmetric InfoNCE between two views.

    Args:
        z1, z2: (B, D) embeddings from two augmented views
        temperature: softmax temperature (lower = sharper)
        normalize: L2-normalize before computing similarity

    Returns:
        scalar loss, float accuracy
    """
    if normalize:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
    B = z1.shape[0]
    labels = torch.arange(B, device=z1.device)
    sim = z1 @ z2.T / temperature
    loss = F.cross_entropy(sim, labels)
    acc = (sim.argmax(1) == labels).float().mean().item()
    return loss, acc


# ══════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def ce_loss(logits, targets):
    """Cross-entropy classification loss.

    Args:
        logits: (B, C) raw logits
        targets: (B,) class indices

    Returns:
        scalar loss, float accuracy
    """
    loss = F.cross_entropy(logits, targets)
    acc = (logits.argmax(-1) == targets).float().mean().item()
    return loss, acc


def ce_loss_paired(logits1, logits2, targets):
    """Averaged CE over two views.

    Returns:
        scalar loss, float accuracy (from view 1)
    """
    l1 = F.cross_entropy(logits1, targets)
    l2 = F.cross_entropy(logits2, targets)
    acc = (logits1.argmax(-1) == targets).float().mean().item()
    return (l1 + l2) / 2, acc


# ══════════════════════════════════════════════════════════════════
# BRIDGE — patchwork predicts constellation's assignment
# ══════════════════════════════════════════════════════════════════

def bridge_loss(bridge_logits, assign_targets, detach_targets=True):
    """Soft cross-entropy: patchwork predicts constellation's soft assignment.

    One-way teaching: constellation → patchwork.
    Targets are detached so constellation is shaped only by internal losses.

    Args:
        bridge_logits: (B, A) raw logits from bridge head
        assign_targets: (B, A) soft assignment from constellation
        detach_targets: detach targets from graph (default True)

    Returns:
        scalar loss, float accuracy (hard agreement)
    """
    if detach_targets:
        assign_targets = assign_targets.detach()
    loss = -(assign_targets * F.log_softmax(bridge_logits, dim=-1)).sum(-1).mean()
    acc = (bridge_logits.argmax(-1) == assign_targets.argmax(-1)).float().mean().item()
    return loss, acc


def bridge_loss_paired(bridge1, bridge2, assign1, assign2, detach_targets=True):
    """Bridge loss averaged over two views.

    Returns:
        scalar loss, float accuracy (from view 1)
    """
    l1, acc = bridge_loss(bridge1, assign1, detach_targets)
    l2, _ = bridge_loss(bridge2, assign2, detach_targets)
    return (l1 + l2) / 2, acc


# ══════════════════════════════════════════════════════════════════
# ASSIGNMENT — internal constellation self-organization
# ══════════════════════════════════════════════════════════════════

def assign_bce_loss(soft_assign, cos_to_anchors):
    """Assignment crispness: BCE toward hard nearest-anchor target.

    Args:
        soft_assign: (B, A) softmax assignment
        cos_to_anchors: (B, A) cosine similarities to anchors

    Returns:
        scalar loss, float entropy
    """
    nearest = cos_to_anchors.argmax(dim=-1)
    hard = torch.zeros_like(soft_assign)
    hard.scatter_(1, nearest.unsqueeze(1), 1.0)

    with torch.amp.autocast("cuda", enabled=False):
        loss = F.binary_cross_entropy(
            soft_assign.float().clamp(1e-7, 1 - 1e-7),
            hard.float(), reduction='mean')

    entropy = -(soft_assign * soft_assign.clamp(min=1e-8).log()).sum(-1).mean().item()
    return loss, entropy


def assign_nce_loss(assign1, assign2, temperature=0.1):
    """Assignment consistency: NCE across two views.

    Args:
        assign1, assign2: (B, A) soft assignments from two views
        temperature: softmax temperature

    Returns:
        scalar loss, float accuracy
    """
    B = assign1.shape[0]
    labels = torch.arange(B, device=assign1.device)
    sim = assign1 @ assign2.T / temperature
    loss = F.cross_entropy(sim, labels)
    acc = (sim.argmax(1) == labels).float().mean().item()
    return loss, acc


# ══════════════════════════════════════════════════════════════════
# ATTRACTION — embeddings near their anchors
# ══════════════════════════════════════════════════════════════════

def attraction_loss(cos_to_anchors):
    """Pull embeddings toward nearest anchor. Higher cos = closer.

    Args:
        cos_to_anchors: (B, A) cosine similarities

    Returns:
        scalar loss, float mean nearest cosine
    """
    nearest_cos = cos_to_anchors.max(dim=1).values
    loss = (1.0 - nearest_cos).mean()
    return loss, nearest_cos.mean().item()


# ══════════════════════════════════════════════════════════════════
# SPREAD — anchor repulsion
# ══════════════════════════════════════════════════════════════════

def spread_loss(anchors, target_cos=0.0):
    """Repulsion loss keeping anchors spread on S^(d-1).

    Args:
        anchors: (A, D) anchor parameters
        target_cos: cosine threshold (0.0 = orthogonal target)

    Returns:
        scalar loss
    """
    a = F.normalize(anchors, dim=-1)
    sim = a @ a.T
    mask = ~torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
    return F.relu(sim[mask] - target_cos).mean()


# ══════════════════════════════════════════════════════════════════
# kNN — non-differentiable validation metric
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def knn_accuracy(embeddings, targets, k=1):
    """k-NN classification accuracy in embedding space.

    Args:
        embeddings: (N, D) L2-normalized
        targets: (N,) class labels
        k: number of neighbors (1 for simple NN)

    Returns:
        float accuracy
    """
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-1)
    if k == 1:
        nn_idx = sim.argmax(dim=1)
        return (targets[nn_idx] == targets).float().mean().item()
    else:
        _, topk_idx = sim.topk(k, dim=1)
        nn_labels = targets[topk_idx]  # (N, k)
        # Majority vote
        pred = nn_labels.mode(dim=1).values
        return (pred == targets).float().mean().item()


# ══════════════════════════════════════════════════════════════════
# THREE-DOMAIN COMPOUND LOSS
# ══════════════════════════════════════════════════════════════════

def three_domain_loss(output, targets, constellation, cv_target=0.22,
                      infonce_temp=0.07, assign_temp=0.1,
                      w_ce=1.0, w_nce_emb=0.5,
                      w_nce_pw=1.0, w_bridge=1.0,
                      w_assign=0.5, w_assign_nce=0.25,
                      w_nce_tri=0.5, w_attract=0.25,
                      w_cv=0.01, w_spread=0.01,
                      cv_batched=True):
    """Full three-domain cooperative loss.

    EXTERNAL:   CE + embedding NCE
    GEOMETRIC:  patchwork NCE + bridge
    INTERNAL:   assign BCE + assign NCE + tri NCE + attraction + CV + spread

    Args:
        output: dict from InternalConstellationCore.forward_paired()
        targets: (B,) class labels
        constellation: Constellation module (for anchors)
        cv_target: CV loss target
        infonce_temp: embedding NCE temperature
        assign_temp: assignment NCE / patchwork NCE temperature
        w_*: per-term weights
        cv_batched: use batched CV (default True)

    Returns:
        total_loss: scalar tensor
        ld: dict with all per-term values and diagnostics
    """
    ld = {}
    emb1, emb2 = output['embedding'], output['embedding_aug']
    B = emb1.shape[0]
    device = emb1.device

    # ── EXTERNAL ──
    l_ce, acc = ce_loss_paired(output['logits'], output['logits_aug'], targets)
    ld['ce'], ld['acc'] = l_ce, acc

    l_nce_emb, nce_emb_acc = nce_loss(emb1, emb2, infonce_temp, normalize=False)
    ld['nce_emb'], ld['nce_emb_acc'] = l_nce_emb, nce_emb_acc

    # ── GEOMETRIC ──
    l_nce_pw, nce_pw_acc = nce_loss(output['patchwork1'], output['patchwork1_aug'],
                                     assign_temp, normalize=True)
    ld['nce_pw'], ld['nce_pw_acc'] = l_nce_pw, nce_pw_acc

    l_bridge, bridge_acc = bridge_loss_paired(
        output['bridge1'], output['bridge2'],
        output['assign1'], output['assign2'])
    ld['bridge'], ld['bridge_acc'] = l_bridge, bridge_acc

    # ── INTERNAL ──
    l_assign, assign_ent = assign_bce_loss(output['assign1'], output['cos1'])
    ld['assign'], ld['assign_entropy'] = l_assign, assign_ent

    l_assign_nce, assign_nce_acc = assign_nce_loss(
        output['assign1'], output['assign2'], assign_temp)
    ld['assign_nce'], ld['assign_nce_acc'] = l_assign_nce, assign_nce_acc

    l_nce_tri, nce_tri_acc = nce_loss(output['tri1'], output['tri2'], 0.1, normalize=True)
    ld['nce_tri'], ld['nce_tri_acc'] = l_nce_tri, nce_tri_acc

    l_attract, nearest_cos = attraction_loss(output['cos1'])
    ld['attract'], ld['nearest_cos'] = l_attract, nearest_cos

    l_cv = cv_loss(emb1, target=cv_target, batched=cv_batched)
    ld['cv'] = l_cv

    l_spread = spread_loss(constellation.anchors)
    ld['spread'] = l_spread

    # ── kNN (non-differentiable) ──
    ld['knn_acc'] = knn_accuracy(emb1, targets)

    # ── TOTAL ──
    loss_external = w_ce * l_ce + w_nce_emb * l_nce_emb
    loss_geometric = w_nce_pw * l_nce_pw + w_bridge * l_bridge
    loss_internal = (w_assign * l_assign + w_assign_nce * l_assign_nce
                     + w_nce_tri * l_nce_tri + w_attract * l_attract
                     + w_cv * l_cv + w_spread * l_spread)

    loss = loss_external + loss_geometric + loss_internal

    ld['loss_external'] = loss_external.item()
    ld['loss_geometric'] = loss_geometric.item()
    ld['loss_internal'] = loss_internal.item()
    ld['total'] = loss

    # Per-term raw values for analysis
    ld['t_ce'] = l_ce.item()
    ld['t_nce_emb'] = l_nce_emb.item()
    ld['t_nce_pw'] = l_nce_pw.item()
    ld['t_bridge'] = l_bridge.item()
    ld['t_assign'] = l_assign.item()
    ld['t_assign_nce'] = l_assign_nce.item()
    ld['t_nce_tri'] = l_nce_tri.item()
    ld['t_attract'] = l_attract.item()

    return loss, ld