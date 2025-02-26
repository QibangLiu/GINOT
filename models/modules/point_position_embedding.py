
"""
Copyright © Qibang Liu 2025. All Rights Reserved.

Author: Qibang Liu <qibang@illinois.edu>
National Center for Supercomputing Applications,
University of Illinois at Urbana-Champaign
Created: 2025-01-15

Based on https://github.com/openai/shap-e/blob/main/shap_e/models/nn/encoding.py
"""

# %%
import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn


# %%


def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """
    Concatenate x and its positional encodings, following NeRF.

    Reference: https://arxiv.org/pdf/2210.04628.pdf
    """
    if min_deg == max_deg:
        return x
    scales = get_scales(min_deg, max_deg, x.dtype, x.device)
    *shape, dim = x.shape
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    return torch.cat([x, emb], dim=-1)


@lru_cache
def get_scales(
    min_deg: int,
    max_deg: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return 2.0 ** torch.arange(min_deg, max_deg, device=device, dtype=dtype)


def encode_position(version: str, *, position: torch.Tensor):
    if version == "v1":
        freqs = get_scales(0, 10, position.dtype, position.device).view(1, -1)
        freqs = position.reshape(-1, 1) * freqs
        return torch.cat([freqs.cos(), freqs.sin()], dim=1).reshape(*position.shape[:-1], -1)
    elif version == "nerf":
        return posenc_nerf(position, min_deg=0, max_deg=15)
    else:
        raise ValueError(version)


def position_encoding_channels(version: Optional[str] = None) -> int:
    if version is None:
        return 1
    return encode_position(version, position=torch.zeros(1, 1)).shape[-1]


class PosEmbLinear(nn.Linear):
    def __init__(
        self, posemb_version: Optional[str], in_features: int, out_features: int, **kwargs
    ):
        super().__init__(
            in_features * position_encoding_channels(posemb_version),
            out_features,
            **kwargs,
        )
        self.posemb_version = posemb_version

    def forward(self, x: torch.Tensor):
        if self.posemb_version is not None:
            x = encode_position(self.posemb_version, position=x)
        return super().forward(x)

# %%
