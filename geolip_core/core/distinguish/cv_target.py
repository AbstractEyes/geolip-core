"""
CV Target Computation & Soft Hand Loss
========================================
Runtime utilities for the GeoLIP loss spectrum.

compute_target_cv(V, D) — Calculate the expected CV for a (V, D) embedding.
    Generates random unit vectors on S^(D-1), measures pentachoron CV.
    This IS the attractor value. No sweep file needed.

cv_proximity(measured, target, sigma) — Gaussian proximity to target CV.
    1.0 at target, decays with distance. Used for soft hand weighting.

soft_hand_weights(proximity, boost, penalty_weight) — Oscillatory counterweight.
    Returns (recon_weight, cv_penalty_weight) for loss composition.

These integrate with the existing cv_loss and cv_metric in losses.py.

Usage:
    from geolip_core.core.distinguish.losses import cv_loss, cv_metric
    from geolip_core.core.distinguish.cv_target import compute_target_cv, cv_proximity, soft_hand_weights

    # Get the attractor for your architecture
    target = compute_target_cv(V=96, D=24)  # ~0.2992

    # During training
    measured = cv_metric(embeddings)
    prox = cv_proximity(measured, target)
    recon_w, cv_w = soft_hand_weights(prox)
    loss = recon_w * recon_loss + cv_w * cv_loss(embeddings, target=target)
"""

import math
import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# CV TARGET COMPUTATION
# ══════════════════════════════════════════════════════════════════

def _pentachoron_cv_single(points, n_samples=200, n_points=5):
    """Compute CV of pentachoron volumes for a single set of points.
    points: (V, D) tensor on S^(D-1).
    Returns: float CV value.
    """
    V, D = points.shape
    device = points.device

    if V < n_points:
        return 0.0

    pool = min(V, 512)
    indices = torch.rand(n_samples, pool, device=device).argsort(dim=1)[:, :n_points]
    pts = points[:pool][indices]  # (n_samples, n_points, D)

    # Cayley-Menger determinant in fp64
    pts_d = pts.double()
    gram = torch.bmm(pts_d, pts_d.transpose(1, 2))
    norms = torch.diagonal(gram, dim1=1, dim2=2)
    d2 = F.relu(norms.unsqueeze(2) + norms.unsqueeze(1) - 2 * gram)

    M = n_points + 1
    cm = torch.zeros(n_samples, M, M, device=device, dtype=torch.float64)
    cm[:, 0, 1:] = 1.0
    cm[:, 1:, 0] = 1.0
    cm[:, 1:, 1:] = d2

    k = n_points - 1
    pf = ((-1.0) ** (k + 1)) / ((2.0 ** k) * (math.factorial(k) ** 2))
    dets = pf * torch.linalg.det(cm)

    valid = dets > 1e-20
    if valid.sum() < 10:
        return 0.0

    vols = dets[valid].sqrt()
    return (vols.std() / (vols.mean() + 1e-8)).item()


@torch.no_grad()
def compute_target_cv(V, D, n_trials=20, n_samples=200, device=None):
    """Compute the expected CV for a (V, D) embedding on S^(D-1).

    Generates random unit vectors and measures their pentachoron CV.
    This is the geometric attractor — the value the system returns to
    after catastrophe, the equilibrium of the weight manifold.

    Args:
        V: Number of embedding rows (vocabulary size)
        D: Embedding dimension
        n_trials: Number of random trials to average (default 20)
        n_samples: Pentachoron samples per trial (default 200)
        device: torch device (default: cuda if available)

    Returns:
        float: Expected CV for (V, D). Typical range 0.13-0.37.

    Example:
        #>>> compute_target_cv(96, 24)    # ~0.2992
        #>>> compute_target_cv(1024, 24)  # ~0.2916
        #>>> compute_target_cv(48, 24)    # ~0.3668
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cvs = []
    for _ in range(n_trials):
        # Random unit vectors on S^(D-1)
        points = torch.randn(V, D, device=device)
        points = F.normalize(points, dim=-1)
        cv = _pentachoron_cv_single(points, n_samples=n_samples)
        if cv > 0:
            cvs.append(cv)

    if not cvs:
        return 0.0

    return sum(cvs) / len(cvs)


@torch.no_grad()
def compute_target_cv_with_stats(V, D, n_trials=50, n_samples=200, device=None):
    """Like compute_target_cv but returns mean, std, and all samples.

    Useful for validating a (V, D) pair and understanding variance.

    Returns:
        dict: {
            'mean': float,     # target CV
            'std': float,      # trial-to-trial variance
            'min': float,
            'max': float,
            'n_valid': int,    # trials that produced valid CV
            'samples': list,   # all individual CV values
        }
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cvs = []
    for _ in range(n_trials):
        points = torch.randn(V, D, device=device)
        points = F.normalize(points, dim=-1)
        cv = _pentachoron_cv_single(points, n_samples=n_samples)
        if cv > 0:
            cvs.append(cv)

    if not cvs:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'n_valid': 0, 'samples': []}

    t = torch.tensor(cvs)
    return {
        'mean': t.mean().item(),
        'std': t.std().item(),
        'min': t.min().item(),
        'max': t.max().item(),
        'n_valid': len(cvs),
        'samples': cvs,
    }


# ══════════════════════════════════════════════════════════════════
# CV PROXIMITY — Gaussian distance to target
# ══════════════════════════════════════════════════════════════════

def cv_proximity(measured_cv, target_cv, sigma=0.15):
    """Gaussian proximity of measured CV to target.

    Returns 1.0 when measured == target.
    Decays smoothly with distance.

    Args:
        measured_cv: float, current measured CV
        target_cv: float, attractor CV for this (V, D)
        sigma: float, transition width (default 0.15)
            At 1 sigma: proximity = 0.61
            At 2 sigma: proximity = 0.14
            At 3 sigma: proximity = 0.01

    Returns:
        float: proximity in [0, 1]
    """
    delta = measured_cv - target_cv
    return math.exp(-delta ** 2 / (2 * sigma ** 2))


# ══════════════════════════════════════════════════════════════════
# SOFT HAND — Oscillatory Counterweight
# ══════════════════════════════════════════════════════════════════

def soft_hand_weights(proximity, boost=0.5, penalty_weight=0.3):
    """Compute loss weights from CV proximity.

    Near target (high proximity):
        - Reconstruction boosted (positive momentum)
        - CV penalty near zero (don't interfere)

    Far from target (low proximity):
        - Reconstruction at baseline (no boost)
        - CV penalty active (restoring force)

    Args:
        proximity: float in [0, 1] from cv_proximity()
        boost: float, max recon multiplier bonus (default 0.5 = 1.5x max)
        penalty_weight: float, max CV penalty weight (default 0.3)

    Returns:
        tuple: (recon_weight, cv_penalty_weight)
            recon_weight: 1.0 to 1.0+boost
            cv_penalty_weight: 0.0 to penalty_weight

    Example:
        prox = cv_proximity(measured_cv=0.30, target_cv=0.2992)  # ~1.0
        rw, cw = soft_hand_weights(prox)  # (1.5, 0.0)

        prox = cv_proximity(measured_cv=0.80, target_cv=0.2992)  # ~0.0
        rw, cw = soft_hand_weights(prox)  # (1.0, 0.3)
    """
    recon_w = 1.0 + boost * proximity
    cv_w = penalty_weight * (1.0 - proximity)
    return recon_w, cv_w


def soft_hand_loss(recon_loss, cv_loss_val, measured_cv, target_cv,
                   boost=0.5, penalty_weight=0.3, sigma=0.15):
    """Complete soft hand loss computation.

    Convenience function that computes proximity and applies weights.

    Args:
        recon_loss: tensor, reconstruction loss (differentiable)
        cv_loss_val: tensor, CV loss (differentiable)
        measured_cv: float, current measured CV (from cv_metric, no grad)
        target_cv: float, attractor CV for this (V, D)
        boost: float, max recon multiplier bonus
        penalty_weight: float, max CV penalty weight
        sigma: float, proximity transition width

    Returns:
        tuple: (weighted_loss, proximity, recon_weight)
    """
    prox = cv_proximity(measured_cv, target_cv, sigma)
    recon_w, cv_w = soft_hand_weights(prox, boost, penalty_weight)
    loss = recon_w * recon_loss + cv_w * cv_loss_val
    return loss, prox, recon_w


# ══════════════════════════════════════════════════════════════════
# INLINE TESTS
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CV Target Computation — validation on {device}")
    print()

    # Test compute_target_cv for known configurations
    test_cases = [
        (96, 24, 0.2992, 0.05),   # validated from sweep
        (48, 24, 0.3668, 0.05),   # validated from sweep
        (1024, 24, 0.2916, 0.05), # validated from sweep
        (200, 24, 0.2914, 0.05),  # validated from sweep
    ]

    print("compute_target_cv validation:")
    for V, D, expected, tol in test_cases:
        result = compute_target_cv_with_stats(V, D, n_trials=30, device=device)
        delta = abs(result['mean'] - expected)
        status = "PASS" if delta < tol else "FAIL"
        print(f"  [{status}] V={V:>4}, D={D}: "
              f"computed={result['mean']:.4f} +/- {result['std']:.4f}, "
              f"expected={expected:.4f}, delta={delta:.4f}")

    # Test proximity
    print()
    print("cv_proximity:")
    for measured, target in [(0.30, 0.2992), (0.50, 0.2992), (0.80, 0.2992), (0.15, 0.2992)]:
        p = cv_proximity(measured, target)
        print(f"  measured={measured:.2f}, target={target:.4f} -> proximity={p:.4f}")

    # Test soft hand weights
    print()
    print("soft_hand_weights:")
    for prox in [1.0, 0.8, 0.5, 0.1, 0.0]:
        rw, cw = soft_hand_weights(prox)
        print(f"  proximity={prox:.1f} -> recon_w={rw:.2f}, cv_w={cw:.3f}")

    print()
    print("Done.")