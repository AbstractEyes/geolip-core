"""
GeometricAttention — attention with FiLM from curated constellation.
Stream B: FiLM on Q,K ONLY — geometry routes attention, V stays pure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geolip_core.pipeline.observer import TorchComponent
from .context_film import FiLMLayer


class GeometricAttention(TorchComponent):
    """Attention with FiLM from curated constellation. Stream B."""
    def __init__(self, name, d_model, n_heads=8, context_dim=128, dropout=0.1):
        super().__init__(name)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.film_q = FiLMLayer(f'{name}_film_q', d_model, context_dim)
        self.film_k = FiLMLayer(f'{name}_film_k', d_model, context_dim)
        self.norm = nn.LayerNorm(d_model)

        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.film_ffn = FiLMLayer(f'{name}_film_ffn', d_model * 4, context_dim)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, geo_ctx, attn_mask=None, key_padding_mask=None):
        B, L, D = x.shape
        H, HD = self.n_heads, self.head_dim

        Q = self.film_q(self.w_q(x), geo_ctx)
        K = self.film_k(self.w_k(x), geo_ctx)
        V = self.w_v(x)

        Q = Q.view(B, L, H, HD).transpose(1, 2)
        K = K.view(B, L, H, HD).transpose(1, 2)
        V = V.view(B, L, H, HD).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_out = (self.dropout(F.softmax(scores, dim=-1)) @ V)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = self.norm(x + self.w_o(attn_out))

        h = F.gelu(self.ffn1(x))
        h = self.film_ffn(h, geo_ctx)
        x = self.ffn_norm(x + self.ffn_drop(self.ffn2(h)))
        return x
