"""
ContentAttention — standard self-attention, Stream A.
No geometric conditioning.
"""

import torch.nn as nn
from geolip_core.pipeline.observer import TorchComponent


class ContentAttention(TorchComponent):
    """Standard self-attention. Stream A. No geometric conditioning."""
    def __init__(self, name, d_model, n_heads=8, dropout=0.1):
        super().__init__(name)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = self.norm(x + a)
        x = self.ffn_norm(x + self.ffn(x))
        return x
