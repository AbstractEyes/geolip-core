"""
Memory — embedding buffers and state management.

Accumulation buffers for AnchorPush, embedding banks for contrastive
learning, and state tracking across training steps. These support
the observation loop but are not a stage themselves.

Usage:
    from geolip_core.core.memory import EmbeddingBuffer
"""

import torch
import torch.nn.functional as F


class EmbeddingBuffer:
    """Fixed-size FIFO buffer for accumulating embeddings + labels.

    Used by AnchorPush to collect a representative sample across
    multiple batches before executing a push step.

    Args:
        capacity: maximum number of embeddings to store
        dim: embedding dimension
        device: torch device
    """

    def __init__(self, capacity, dim, device='cpu'):
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.embeddings = torch.empty(0, dim, device=device)
        self.labels = torch.empty(0, dtype=torch.long, device=device)

    def add(self, emb, labels):
        """Add a batch of embeddings + labels. Drops oldest if over capacity.

        Args:
            emb: (B, D) embeddings (will be detached and L2-normalized)
            labels: (B,) class indices
        """
        emb = F.normalize(emb.detach(), dim=-1).to(self.device)
        labels = labels.detach().to(self.device)
        self.embeddings = torch.cat([self.embeddings, emb], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        if self.embeddings.shape[0] > self.capacity:
            self.embeddings = self.embeddings[-self.capacity:]
            self.labels = self.labels[-self.capacity:]

    def get(self):
        """Return current (embeddings, labels) tensors."""
        return self.embeddings, self.labels

    def clear(self):
        """Reset buffer to empty."""
        self.embeddings = torch.empty(0, self.dim, device=self.device)
        self.labels = torch.empty(0, dtype=torch.long, device=self.device)

    @property
    def size(self):
        return self.embeddings.shape[0]

    @property
    def full(self):
        return self.embeddings.shape[0] >= self.capacity

    def __len__(self):
        return self.embeddings.shape[0]