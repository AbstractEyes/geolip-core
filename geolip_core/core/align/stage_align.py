"""
StageAligner -- Procrustes-align constellations between training stages.

Given Stage K-1's constellation and Stage K's embedding space:
- Procrustes-align old anchors into new space
- Identify stable anchors (low residual after rotation)
- Identify drifting anchors (high residual)
- Identify dead anchors (no samples assigned)
- Reassign dead anchors to confusion midpoints or repulsion positions

Usage:
    aligner = StageAligner(manifold_dim=256)
    aligned_constellation, report = aligner.align(
        old_constellation, new_embeddings, new_labels)
"""

import torch
import torch.nn.functional as F

from geolip_core.core.align.procrustes import ProcrustesAlignment
from geolip_core.core.distinguish.losses import cv_metric


class StageAligner:
    """Procrustes-align constellation from previous stage to new embedding space.

    Args:
        manifold_dim: embedding dimension
        rank: Procrustes projection rank (for dim > 32)
        whiten: apply Newton-Schulz whitening in Procrustes
        stable_threshold: max residual for an anchor to be considered stable
        min_util: minimum number of assigned samples for an anchor to be alive
    """

    def __init__(self, manifold_dim, rank=24, whiten=True,
                 stable_threshold=0.1, min_util=5):
        self.manifold_dim = manifold_dim
        self.rank = rank
        self.whiten = whiten
        self.stable_threshold = stable_threshold
        self.min_util = min_util
        self._aligner = ProcrustesAlignment(
            dim=manifold_dim, rank=rank, whiten=whiten)

    @torch.no_grad()
    def align(self, old_constellation, new_embeddings, new_labels):
        """Align old constellation anchors to new embedding space.

        Args:
            old_constellation: Constellation module from previous stage
            new_embeddings: (N, D) normalized embeddings from current stage
            new_labels: (N,) labels

        Returns:
            aligned_anchors: (A, D) rotated anchors in new space
            report: dict with alignment diagnostics
        """
        old_anchors = F.normalize(old_constellation.anchors.data, dim=-1)
        A = old_anchors.shape[0]

        # Compute new class centroids
        classes = new_labels.unique(sorted=True)
        new_centroids = []
        for c in classes:
            mask = new_labels == c
            if mask.sum() > 0:
                new_centroids.append(
                    F.normalize(new_embeddings[mask].mean(dim=0), dim=0))
        new_centroids = torch.stack(new_centroids, dim=0)  # (C, D)

        # Find corresponding old anchors for each centroid (nearest)
        cos_sim = old_anchors @ new_centroids.T  # (A, C)
        anchor_to_centroid = cos_sim.argmax(dim=1)  # (A,)
        centroid_to_anchor = cos_sim.argmax(dim=0)  # (C,)

        # Build source-target pairs for Procrustes
        # Use the centroid-matched anchor pairs
        n_pairs = min(A, new_centroids.shape[0])
        source_pts = old_anchors[centroid_to_anchor[:n_pairs]]  # (n_pairs, D)
        target_pts = new_centroids[:n_pairs]  # (n_pairs, D)

        # Procrustes alignment
        aligned_pts, info = self._aligner(
            source_pts.unsqueeze(0), target_pts.unsqueeze(0))

        # Apply the rotation to ALL old anchors
        if 'rotation' in info:
            R = info['rotation'].squeeze(0)  # (D, D)
        elif 'rotation_k' in info:
            R = info['rotation_k'].squeeze(0)
        else:
            R = torch.eye(self.manifold_dim, device=old_anchors.device)

        aligned_anchors = F.normalize(old_anchors @ R.T, dim=-1)

        # Classify anchors
        classification = self.classify_anchors(
            aligned_anchors, new_centroids, new_embeddings)

        # Compute CV
        cv = cv_metric(aligned_anchors.unsqueeze(0).expand(
            64, -1, -1).reshape(-1, self.manifold_dim))

        report = {
            'cos_after': info.get('cos_after', 0.0),
            'n_stable': classification['n_stable'],
            'n_drifting': classification['n_drifting'],
            'n_dead': classification['n_dead'],
            'stable_mask': classification['stable_mask'],
            'drifting_mask': classification['drifting_mask'],
            'dead_mask': classification['dead_mask'],
            'residuals': classification['residuals'],
            'utilization': classification['utilization'],
            'cv': cv,
            'cv_in_band': 0.20 <= cv <= 0.23,
        }

        return aligned_anchors, report

    @torch.no_grad()
    def classify_anchors(self, aligned_anchors, new_centroids, new_embeddings):
        """Classify each anchor as stable, drifting, or dead.

        Args:
            aligned_anchors: (A, D) rotated anchors
            new_centroids: (C, D) class centroids in new space
            new_embeddings: (N, D) all embeddings

        Returns:
            dict with stable_mask, drifting_mask, dead_mask, residuals, utilization
        """
        A = aligned_anchors.shape[0]

        # Residual: distance from aligned anchor to its nearest centroid
        cos_to_centroids = aligned_anchors @ new_centroids.T  # (A, C)
        nearest_cos = cos_to_centroids.max(dim=1).values  # (A,)
        residuals = 1.0 - nearest_cos  # lower = more stable

        # Utilization: how many embeddings are closest to each anchor
        cos_to_emb = new_embeddings @ aligned_anchors.T  # (N, A)
        nearest_anchor = cos_to_emb.argmax(dim=1)  # (N,)
        utilization = torch.bincount(nearest_anchor, minlength=A).float()

        # Classification
        stable_mask = (residuals < self.stable_threshold) & (utilization >= self.min_util)
        dead_mask = utilization < self.min_util
        drifting_mask = ~stable_mask & ~dead_mask

        return {
            'stable_mask': stable_mask,
            'drifting_mask': drifting_mask,
            'dead_mask': dead_mask,
            'n_stable': stable_mask.sum().item(),
            'n_drifting': drifting_mask.sum().item(),
            'n_dead': dead_mask.sum().item(),
            'residuals': residuals,
            'utilization': utilization,
        }

    @torch.no_grad()
    def reassign_dead(self, aligned_anchors, dead_mask, new_centroids, confusion=None):
        """Reposition dead anchors at confusion midpoints or random positions.

        Args:
            aligned_anchors: (A, D) current aligned anchors
            dead_mask: (A,) boolean mask of dead anchors
            new_centroids: (C, D) class centroids
            confusion: (C, C) optional confusion matrix for midpoint placement

        Returns:
            updated_anchors: (A, D) with dead anchors repositioned
        """
        updated = aligned_anchors.clone()
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        n_dead = dead_idx.shape[0]

        if n_dead == 0:
            return updated

        if confusion is not None and confusion.sum() > 0:
            # Place at confusion midpoints
            C = new_centroids.shape[0]
            idx_i, idx_j = torch.triu_indices(C, C, offset=1)
            pair_confusion = confusion[idx_i, idx_j]
            _, sorted_pairs = pair_confusion.sort(descending=True)

            for k in range(min(n_dead, len(sorted_pairs))):
                p = sorted_pairs[k]
                i, j = idx_i[p].item(), idx_j[p].item()
                midpoint = F.normalize(
                    new_centroids[i] + new_centroids[j], dim=0)
                updated[dead_idx[k]] = midpoint

            # Any remaining dead anchors: random
            if n_dead > len(sorted_pairs):
                for k in range(len(sorted_pairs), n_dead):
                    updated[dead_idx[k]] = F.normalize(
                        torch.randn(self.manifold_dim, device=updated.device), dim=0)
        else:
            # No confusion data: random placement
            for k in range(n_dead):
                updated[dead_idx[k]] = F.normalize(
                    torch.randn(self.manifold_dim, device=updated.device), dim=0)

        return F.normalize(updated, dim=-1)

    @torch.no_grad()
    def structural_report(self, old_anchors, new_anchors):
        """SVD of drift vectors and eigenspectrum comparison.

        Args:
            old_anchors: (A, D) anchors before alignment
            new_anchors: (A, D) anchors after alignment

        Returns:
            dict with drift statistics
        """
        drift = new_anchors - old_anchors
        drift_norms = drift.norm(dim=-1)

        # SVD of drift matrix
        U, S, Vh = torch.linalg.svd(drift, full_matrices=False)
        explained = S ** 2
        total_var = explained.sum()
        cumulative = explained.cumsum(0) / total_var.clamp(min=1e-12)

        return {
            'mean_drift': drift_norms.mean().item(),
            'max_drift': drift_norms.max().item(),
            'min_drift': drift_norms.min().item(),
            'drift_std': drift_norms.std().item(),
            'top_singular_values': S[:5].tolist(),
            'variance_explained_top5': cumulative[:5].tolist() if len(cumulative) >= 5 else cumulative.tolist(),
        }
