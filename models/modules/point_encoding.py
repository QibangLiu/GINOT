"""
Copyright © Qibang Liu 2025. All Rights Reserved.

Author: Qibang Liu <qibang@illinois.edu>
National Center for Supercomputing Applications,
University of Illinois at Urbana-Champaign
Created: 2025-01-15

Based on https://github.com/openai/shap-e/blob/main/shap_e/models/nn/ops.py

"""

# %%

import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .transformer import ResidualCrossAttentionBlock
from .pointnet2_utils import sample_and_group, sample_and_group_all
import warnings

# %%


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def geglu(x):
    v, gates = x.chunk(2, dim=-1)
    return v * torch.nn.functional.gelu(gates)


class SirenSin:
    def __init__(self, w0=30.0):
        self.w0 = w0

    def __call__(self, x):
        return torch.sin(self.w0 * x)


def get_act(name):
    return {
        "relu": torch.nn.functional.relu,
        "leaky_relu": torch.nn.functional.leaky_relu,
        "swish": torch.nn.functional.silu,
        "tanh": torch.tanh,
        "quick_gelu": quick_gelu,
        "torch_gelu": torch.nn.functional.gelu,
        "gelu2": quick_gelu,
        "geglu": geglu,
        "sigmoid": torch.sigmoid,
        "sin": torch.sin,
        "sin30": SirenSin(w0=30.0),
        "softplus": torch.nn.functional.softplus,
        "exp": torch.exp,
        "identity": lambda x: x,
    }[name]


class PointSetEmbedding(nn.Module):
    def __init__(
        self,
        *,
        ndim: int,
        radius: float,
        n_point: int,
        n_sample: int,
        d_input: int,
        d_hidden: List[int],
        patch_size: int = 1,
        stride: int = 1,
        activation: Optional[Union[str, nn.Module]] = "swish",
        group_all: bool = False,
        padding_mode: str = "zeros",
        fps_method: str = "first",
        **kwargs,
    ):
        """
        ndim: dimension, = 2 for 2D points, = 3 for 3D points
        """
        super().__init__()
        self.n_point = n_point
        self.radius = radius
        self.n_sample = n_sample
        self.mlp_convs = nn.ModuleList()
        if isinstance(activation, str):
          self.act = get_act(activation)
        else:
          self.act = activation
        self.patch_size = patch_size
        self.stride = stride
        last_channel = d_input + ndim
        for out_channel in d_hidden:
            self.mlp_convs.append(
                nn.Conv2d(
                    last_channel,
                    out_channel,
                    kernel_size=(patch_size, 1),
                    stride=(stride, 1),
                    padding=(patch_size // 2, 0),
                    padding_mode=padding_mode,
                    **kwargs,
                )
            )
            last_channel = out_channel
        self.group_all = group_all
        self.fps_method = fps_method

    def forward(self, xyz, points, pc_padding_value: Optional[int] = None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: sample points feature data, [B, d_hidden[-1], n_point]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            # QB: 2025-01-24
            # we can not use group_all here, since some points are padding points
            # if want group_all, padding points should be avoided
            warnings.warn(
                "group_all can not be used with padding points, I have not implemented it yet")
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.n_point,
                self.radius,
                self.n_sample,
                xyz,
                points,
                deterministic=not self.training,
                fps_method=self.fps_method,
                pc_padding_value=pc_padding_value,
            )
        # new_xyz: sampled points position data, [B, n_point, C]
        # new_points: sampled points data, [B, n_point, n_sample, C+D]
        # [B, C+D, n_sample, n_point]
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            new_points = self.act(self.apply_conv(new_points, conv))

        new_points = new_points.mean(dim=2)
        return new_points

    def apply_conv(self, points: torch.Tensor, conv: nn.Module):
        batch, channels, n_samples, _ = points.shape
        # Shuffle the representations
        if self.patch_size > 1:
            # TODO shuffle deterministically when not self.training
            """
            QB: 2025-01-24
            this part seems has no effect, since a mean pooling is applied after that
            """
            _, indices = torch.rand(
                batch, channels, n_samples, 1, device=points.device).sort(dim=2)
            points = torch.gather(
                points, 2, torch.broadcast_to(indices, points.shape))
        return conv(points)


class SimplePerceiver(nn.Module):
    """
    Only does cross attention
    """

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        layers: int
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=width,
                    heads=heads
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, data: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, data, key_padding_mask=key_padding_mask)
        return x
