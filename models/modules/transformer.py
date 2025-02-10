# %%
import math
from typing import Optional

import torch
import torch.nn as nn


# %%
class MLP(nn.Module):
    def __init__(self, width: int, in_channels: Optional[int] = None, out_channels: Optional[int] = None):
        super().__init__()
        if in_channels is None:
            in_channels = width
        if out_channels is None:
            out_channels = width
        self.width = width
        self.c_fc = nn.Linear(in_channels, width * 4)
        self.c_proj = nn.Linear(width * 4, out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        dropout=0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=heads, batch_first=True, dropout=dropout)
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_3 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)  # Dropout for MLP output

    def forward(self, x: torch.Tensor, data: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        q = self.ln_1(x)
        kv = self.ln_2(data)
        x = x + self.attn(q, kv, kv, key_padding_mask)[0]
        x = x + self.dropout(self.mlp(self.ln_3(x)))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        dropout=0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=heads, batch_first=True, dropout=dropout)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width)
        self.ln_2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)  # Dropout for MLP output

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        qkv = self.ln_1(x)
        x = x + self.attn(qkv, qkv, qkv, key_padding_mask)[0]
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        layers: int,
        dropout=0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width=width,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        return x
