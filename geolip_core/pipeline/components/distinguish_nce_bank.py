"""
GeoResidualBank — cross-stream contrastive memory bank (CLIP-style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoResidualBank(nn.Module):
    """Cross-stream contrastive memory bank (CLIP-style)."""
    def __init__(self, proj_dim, bank_size=4096, temperature=0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.bank_size = bank_size
        self.temperature = temperature

        self.register_buffer('queue', torch.randn(bank_size, proj_dim))
        self.queue = F.normalize(self.queue, dim=-1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys):
        B = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        if ptr + B <= self.bank_size:
            self.queue[ptr:ptr + B] = keys
        else:
            overflow = (ptr + B) - self.bank_size
            self.queue[ptr:] = keys[:B - overflow]
            self.queue[:overflow] = keys[B - overflow:]
        self.queue_ptr[0] = (ptr + B) % self.bank_size

    def forward(self, content_proj, geo_proj):
        q = F.normalize(content_proj, dim=-1)
        k_pos = F.normalize(geo_proj, dim=-1)
        k_neg = self.queue.clone().detach()

        pos_logits = (q * k_pos).sum(dim=-1, keepdim=True) / self.temperature
        neg_logits = q @ k_neg.T / self.temperature

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)

        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == 0).float().mean()

        return loss, acc
