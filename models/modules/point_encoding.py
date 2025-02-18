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

from .transformer import ResidualCrossAttentionBlock, SelfAttentionBlocks
from . import pointnet2_utils as pnet
from .point_position_embedding import PosEmbLinear
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
        # clear the cache
        pnet.CACHE_SAMPLE_AND_GROUP_INDECIES.clear()

    def forward(self, xyz, points, pc_padding_value: Optional[int] = None,
                sample_ids=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            sample_ids: the sample ids in each batch,[B,],
            used for cache the sample and group indices
        Return:
            new_points: sample points feature data, [B, d_hidden[-1], n_point]
        """
        if sample_ids is not None:
            deterministic = True
        else:
            deterministic = not self.training

        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            # QB: 2025-01-24
            # we can not use group_all here, since some points are padding points
            # if want group_all, padding points should be avoided
            warnings.warn(
                "group_all can not be used with padding points, I have not implemented it yet")
            new_xyz, new_points = pnet.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = pnet.sample_and_group(
                self.n_point,
                self.radius,
                self.n_sample,
                xyz,
                points,
                deterministic=deterministic,  # not self.training,
                fps_method=self.fps_method,
                pc_padding_value=pc_padding_value,
                sample_ids=sample_ids
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
        layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=width,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, data: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        for block in self.resblocks:
            x = block(x, data, key_padding_mask=key_padding_mask)
        return x


class PointCloudPerceiverChannelsEncoder(nn.Module):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(self,
                 input_channels: int = 2,
                 out_c: int = 128,
                 width: int = 128,
                 latent_d: int = 128,
                 n_point: int = 128,
                 n_sample: int = 8,
                 radius: float = 0.2,
                 patch_size: int = 8,
                 padding_mode: str = "circular",
                 d_hidden: List[int] = [128, 128],
                 fps_method: str = 'first',
                 num_heads: int = 4,
                 cross_attn_layers: int = 1,
                 self_attn_layers: int = 3,
                 pc_padding_val: Optional[int] = None,
                 dropout: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Args:
            input_channels (int): 2 or 3
            width (int): hidden dimension
            latent_d (int): number of context points
            n_point (int): number of points in the point set embedding
            n_sample (int): number of samples in the point set embedding
            radius (float): radius for the point set embedding
            patch_size//2 (int): padding size of dim 1 of conv in the point set embedding
            padding_mode (str): padding mode of the conv in the point set embedding
            d_hidden (list): hidden dimensions for the conv in the point set embedding
            fps_method (str): method for point sampling in the point set embedding, 'fps' or 'first'
            out_c (int): output channels
            pc_padding_val (int): padding value for the sequence points
            final out shape: [B, out_c*latent_d]
        """
        self.width = width
        self.latent_d = latent_d
        self.n_point = n_point
        self.out_c = out_c
        self.pc_padding_val = pc_padding_val
        self.fps_method = fps_method
        if d_hidden[-1] != self.width:
            warnings.warn(
                "PointCloudPerceiverChannelsEncoder: d_hidden[-1] should be equal to width. d_hidden[-1] is set to width!!!")
            d_hidden[-1] = self.width
        # position embeding + linear layer
        self.pos_emb_linear = PosEmbLinear("nerf", input_channels, self.width)
        d_input = self.width
        self.point_set_embedding \
            = PointSetEmbedding(ndim=input_channels, radius=radius, n_point=self.n_point,
                                n_sample=n_sample, d_input=d_input,
                                d_hidden=d_hidden, patch_size=patch_size,
                                padding_mode=padding_mode,
                                fps_method=fps_method)
        if self.latent_d is not None:
            self.register_parameter(
                "output_tokens",
                nn.Parameter(torch.randn(self.latent_d, self.width)),
            )
        self.ln_pre = nn.LayerNorm(self.width)
        self.ln_post = nn.LayerNorm(self.width)

        self.encoder = SimplePerceiver(
            width=self.width, heads=num_heads, layers=cross_attn_layers, dropout=dropout)
        self.processor = SelfAttentionBlocks(
            width=self.width, heads=num_heads, layers=self_attn_layers, dropout=dropout)
        self.output_proj = nn.Linear(
            self.width, self.out_c)

    def forward(self, points, sample_ids=None, apply_padding_pointnet2=False):
        """
        Args:
            points (torch.Tensor): [B, N, C]
                   C =2 or 3, or >3 if has other features, e.g. rgb color
            sample_ids (torch.Tensor): [B,] the sample ids in each batch,
                     used for cache the sample and group indices.
                     if None, the sampling is not cached, sampling may not be deterministic
                     and resampling is applied in each forward pass.
                     if points number is too large, it is recommended to set sample_ids for efficiency, otherwise,can be None.
            apply_padding_pointnet2 (bool): if True or fps method, set padding value=pc_padding_val for pointnet2
                    otherwise, set padding value=None
        Returns:
            torch.Tensor: [B,latent_d, out_c] if if self.latent_d is not None, else [B,n_point, width]
        """
        # set padding for point set embedding
        if apply_padding_pointnet2 or self.fps_method == "fps":
            # for fps, if padded, must set the padding value for pointnet2
            pc_padding_val_pointnet2 = self.pc_padding_val
        else:
            # for first methods, it depends on the padding value of the input points
            # if the padding value is much larger than radius, it is safe to set it to None
            pc_padding_val_pointnet2 = None

        # set padding for cross-attention
        if self.pc_padding_val is not None:
            pading_mask = points[:, :, 0] == self.pc_padding_val  # [B, N]
            pading_mask = pading_mask.to(points.device)
        else:
            pading_mask = None
        # B,N,C1]-> [B,C1,N]
        xyz = points.permute(0, 2, 1)
        # [B, N, C1] -> [B, N, C2], C2=self.width
        dataset_emb = self.pos_emb_linear(points)  # [B, N, C]
        # [B, N, C2] -> [B, C2, N]
        points = dataset_emb.permute(0, 2, 1)
        # [B, C2, N] -------------> [B, C3, No], No=n_point
        #      \ pointNet             /\ mean (dim=2)
        #      _\/ permute           / Conv, C3=d_hidden[-1]=width
        #       [B, C2+ndim,  n_sample, n_point]
        data_tokens = self.point_set_embedding(
            xyz, points, pc_padding_val_pointnet2,
            sample_ids=sample_ids)
        # [B, Co, No] -> [B, No, Co]
        data_tokens = data_tokens.permute(0, 2, 1)
        batch_size = points.shape[0]
        if self.latent_d is not None:
            latent_tokens = self.output_tokens.unsqueeze(
                0).repeat(batch_size, 1, 1)  # [B, latent_d, width]
            # [B, n_point+latent_d, width]
            h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
            assert h.shape == (batch_size, self.n_point +
                               self.latent_d, self.width)
        else:
            h = self.ln_pre(data_tokens)
        # [B, Nnl, width] -> [B,  Nnl, width], Nnl=n_point+latent_d or n_point
        # cross_attn. TODO: add mask here, dataset_emb has padding points
        h = self.encoder(h, dataset_emb, key_padding_mask=pading_mask)
        # [B,  Nnl, width]-> [B,  Nnl, width]
        h = self.processor(h)
        # [B,  Nnl, width] -> [B, latent_d, width]
        # -> [B, latent_d, out_c]
        if self.latent_d is not None:
            h = h[:, -self.latent_d:]
        h = self.output_proj(self.ln_post(h))
        h = nn.Tanh()(h)  # project to [-1,1]
        # h = h.view(batch_size, -1)
        return h  # [B, latent_d, out_c]
