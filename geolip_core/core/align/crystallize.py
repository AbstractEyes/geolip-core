"""
CrystallizationEngine -- construct constellations from frozen model embeddings.

Episodic crystallization: freeze model, collect embeddings, compute class
centroids on S^(d-1), place anchors at centroids + confusion midpoints,
validate CV is in the pentachoron band (0.20-0.23).

Usage:
    engine = CrystallizationEngine(manifold_dim=256, n_anchors=128, n_classes=100)
    constellation, report = engine.crystallize(model, loader, project_fn)
"""

import torch
import torch.nn.functional as F

from geolip_core.core.associate.constellation import (
    Constellation, init_anchors_repulsion,
)
from geolip_core.core.distinguish.losses import cv_metric


class CrystallizationEngine:
    """Construct constellations from frozen model embeddings.

    Anchor layout:
        slots 0..n_classes-1:      class centroids on S^(d-1)
        slots n_classes..n_classes+n_confusion-1: midpoints between confused pairs
        remaining slots:           repulsion-initialized in gaps

    Args:
        manifold_dim: dimension of the hypersphere embedding
        n_anchors: total number of constellation anchors
        n_classes: number of target classes
    """

    def __init__(self, manifold_dim, n_anchors, n_classes):
        self.manifold_dim = manifold_dim
        self.n_anchors = n_anchors
        self.n_classes = n_classes

    @torch.no_grad()
    def collect_embeddings(self, model, loader, project_fn, max_samples=50000):
        """Run frozen model over data and collect (embedding, label) pairs.

        Args:
            model: frozen model (eval mode)
            loader: data loader yielding (inputs, labels) batches
            project_fn: callable(model, batch_inputs) -> (B, manifold_dim) embeddings
                        The function should handle device placement internally.
            max_samples: cap on number of samples to collect

        Returns:
            embeddings: (N, manifold_dim) on CPU, normalized to S^(d-1)
            labels: (N,) on CPU
        """
        model.eval()
        all_emb, all_lbl = [], []
        n_collected = 0

        for batch in loader:
            if n_collected >= max_samples:
                break
            inputs, labels = batch[0], batch[1]
            emb = project_fn(model, inputs)  # (B, manifold_dim)
            emb = F.normalize(emb, dim=-1)

            remaining = max_samples - n_collected
            if emb.shape[0] > remaining:
                emb = emb[:remaining]
                labels = labels[:remaining]

            all_emb.append(emb.cpu())
            all_lbl.append(labels.cpu())
            n_collected += emb.shape[0]

        return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)

    def compute_centroids(self, embeddings, labels):
        """Per-class mean centroids on S^(d-1).

        Args:
            embeddings: (N, D) normalized embeddings
            labels: (N,) class labels

        Returns:
            centroids: (n_classes, D) normalized centroids
            classes: (n_classes,) unique class labels
        """
        classes = labels.unique(sorted=True)
        centroids = []
        for c in classes:
            mask = labels == c
            if mask.sum() > 0:
                centroid = embeddings[mask].mean(dim=0)
                centroids.append(F.normalize(centroid, dim=0))
            else:
                centroids.append(torch.randn(self.manifold_dim))
        return torch.stack(centroids, dim=0), classes

    def confusion_matrix(self, embeddings, labels, centroids, classes):
        """Compute soft confusion counts between class centroids.

        Args:
            embeddings: (N, D)
            labels: (N,)
            centroids: (C, D)
            classes: (C,) class labels

        Returns:
            confusion: (C, C) symmetric confusion matrix
                       confusion[i,j] = how often class i samples are closer to centroid j
        """
        C = centroids.shape[0]
        # Compute assignment: each sample -> nearest centroid
        cos_sim = embeddings @ centroids.T  # (N, C)
        nearest_centroid = cos_sim.argmax(dim=-1)  # (N,)

        # Build class index mapping
        class_to_idx = {c.item(): i for i, c in enumerate(classes)}

        confusion = torch.zeros(C, C, dtype=torch.float32)
        for i in range(embeddings.shape[0]):
            true_class = class_to_idx.get(labels[i].item(), -1)
            pred_class = nearest_centroid[i].item()
            if true_class >= 0 and true_class != pred_class:
                confusion[true_class, pred_class] += 1
                confusion[pred_class, true_class] += 1

        return confusion

    def place_confusion_anchors(self, centroids, confusion, n_extra):
        """Place anchors at midpoints between most-confused class pairs.

        Args:
            centroids: (C, D) normalized centroids
            confusion: (C, C) confusion matrix
            n_extra: number of confusion anchors to place

        Returns:
            anchors: (n_extra, D) normalized midpoint anchors
        """
        if n_extra <= 0:
            return torch.zeros(0, self.manifold_dim)

        C = centroids.shape[0]
        # Get upper triangle of confusion matrix
        idx_i, idx_j = torch.triu_indices(C, C, offset=1)
        pair_confusion = confusion[idx_i, idx_j]

        # Sort by confusion count (descending)
        _, sorted_idx = pair_confusion.sort(descending=True)
        n_pairs = min(n_extra, len(sorted_idx))

        anchors = []
        for k in range(n_pairs):
            p = sorted_idx[k]
            i, j = idx_i[p].item(), idx_j[p].item()
            midpoint = F.normalize(centroids[i] + centroids[j], dim=0)
            anchors.append(midpoint)

        return torch.stack(anchors, dim=0)

    def build_constellation(self, anchor_positions, device=None):
        """Create a Constellation with pre-placed anchor positions.

        Args:
            anchor_positions: (n_anchors, manifold_dim) pre-placed positions
            device: target device

        Returns:
            constellation: Constellation module with anchors set
        """
        constellation = Constellation(
            dim=self.manifold_dim,
            n_anchors=self.n_anchors,
        )
        with torch.no_grad():
            constellation.anchors.data.copy_(anchor_positions)
        if device is not None:
            constellation = constellation.to(device)
        return constellation

    def crystallize(self, model, loader, project_fn, device=None, max_samples=50000):
        """Full crystallization pipeline.

        Args:
            model: frozen model
            loader: data loader
            project_fn: callable(model, inputs) -> (B, manifold_dim)
            device: target device for the constellation
            max_samples: cap on embedding collection

        Returns:
            constellation: Constellation with crystallized anchors
            report: dict with diagnostics
        """
        # 1. Collect embeddings
        embeddings, labels = self.collect_embeddings(
            model, loader, project_fn, max_samples=max_samples)

        # 2. Compute class centroids
        centroids, classes = self.compute_centroids(embeddings, labels)
        n_centroid = centroids.shape[0]

        # 3. Compute confusion matrix
        confusion = self.confusion_matrix(embeddings, labels, centroids, classes)

        # 4. Place confusion anchors
        n_confusion = min(self.n_anchors - n_centroid, n_centroid * (n_centroid - 1) // 2)
        n_confusion = max(0, n_confusion)
        confusion_anchors = self.place_confusion_anchors(
            centroids, confusion, n_confusion)

        # 5. Fill remaining with repulsion-initialized anchors
        n_placed = n_centroid + confusion_anchors.shape[0]
        n_repulsion = self.n_anchors - n_placed

        if n_repulsion > 0:
            # Initialize remaining anchors via repulsion in the gaps
            remaining = init_anchors_repulsion(
                n_repulsion, self.manifold_dim, n_iter=200)
        else:
            remaining = torch.zeros(0, self.manifold_dim)

        # 6. Assemble all anchors
        parts = [centroids]
        if confusion_anchors.shape[0] > 0:
            parts.append(confusion_anchors)
        if remaining.shape[0] > 0:
            parts.append(remaining)
        all_anchors = F.normalize(torch.cat(parts, dim=0), dim=-1)

        # Ensure exact n_anchors
        if all_anchors.shape[0] > self.n_anchors:
            all_anchors = all_anchors[:self.n_anchors]
        elif all_anchors.shape[0] < self.n_anchors:
            # Pad with random normalized anchors
            pad = F.normalize(
                torch.randn(self.n_anchors - all_anchors.shape[0], self.manifold_dim),
                dim=-1)
            all_anchors = torch.cat([all_anchors, pad], dim=0)

        # 7. Build constellation
        constellation = self.build_constellation(all_anchors, device=device)

        # 8. Validate
        anchors_n = F.normalize(constellation.anchors.data, dim=-1)
        cv = cv_metric(anchors_n.unsqueeze(0).expand(64, -1, -1).reshape(-1, self.manifold_dim))

        # Centroid spread
        if n_centroid > 1:
            cos = centroids @ centroids.T
            idx = torch.triu_indices(n_centroid, n_centroid, offset=1)
            pairwise_cos = cos[idx[0], idx[1]]
            centroid_spread = pairwise_cos.mean().item()
        else:
            centroid_spread = 0.0

        report = {
            'cv': cv,
            'n_samples_collected': embeddings.shape[0],
            'n_classes_found': n_centroid,
            'n_centroid_anchors': n_centroid,
            'n_confusion_anchors': confusion_anchors.shape[0],
            'n_repulsion_anchors': max(0, n_repulsion),
            'centroid_spread': centroid_spread,
            'cv_in_band': 0.20 <= cv <= 0.23,
        }

        return constellation, report
