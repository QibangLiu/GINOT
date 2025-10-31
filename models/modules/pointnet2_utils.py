"""
Author: Qibang Liu <qibang@illinois.edu>
National Center for Supercomputing Applications,
University of Illinois at Urbana-Champaign
Created: 2025-01-15

Modified base on https://github.com/openai/shap-e/blob/main/shap_e/models/nn/pointnet2_utils.py
Originaly based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
MIT License
Copyright (c) 2019 benny
"""

from typing import Optional
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# have to clear before using the cache
CACHE_SAMPLE_AND_GROUP_INDECIES = {}


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S], S is number of points to be selected out
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)  # [B,S], e.g. [B,128]
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [B,1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # [1,S], e.g.[1,128]
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(
            device).view(view_shape).repeat(repeat_shape)
    )  # shape: [B,S]
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint,
                          pc_padding_value: Optional[float] = None,
                          deterministic=False,
                          order_invariant: bool = False):
    """
    Input:
        xyz: pointcloud data, [B, N, ndim], ndim=3 or 2
        npoint: number of samples
        pc_padding_value: padding value in point cloud, if not None, the padding value will not be sampled
        order_invariant:
            If True, always use the smallest point as first point of fps, ensuring that the output remains identical even if the input point cloud order is shuffled.
            If False, final results may vary slightly for shuffled inputs, but the differences are typically minor and negligible.
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape  # C is ndim
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if pc_padding_value is not None:
        """
            avoid sampling the padding points
            for the following farthest point sampling
        """
        pad_mask = xyz[:, :, 0] == pc_padding_value  # [B, N]
        distance[pad_mask] = -1

    if not order_invariant:
        if pc_padding_value is not None:
            # QB if padding_value and the points is shuffled
            """
            avoid sampling the padding points
            for the following farthest point sampling
            """
            non_pad_idx = torch.arange(N).repeat(B, 1).to(device)
            non_pad_idx = torch.where(~pad_mask, non_pad_idx, float('inf'))
            non_pad_idx, _ = torch.sort(non_pad_idx, dim=1)
            non_pad_idx = non_pad_idx[:, :npoint].long().to(device)
            """avoid sampling the padding points for the first point"""
            if deterministic:
                farthest = non_pad_idx[:, 0]
            else:
                ids = torch.randint(0, npoint, (B,), dtype=torch.long)
                farthest = non_pad_idx[torch.arange(B), ids]
        else:
            if deterministic:
                # farthest = torch.arange(0, B, dtype=torch.long).to(device)
                farthest = torch.zeros(B, dtype=torch.long).to(device)
            else:
                farthest = torch.randint(
                    0, N, (B,), dtype=torch.long).to(device)
    else:
        """ use the smallest x point as the first point for fps to ensure order_invariant"""
        # x_clone = xyz.clone()  # [B, N, C]
        # if pc_padding_value is not None:
        #     x_clone[x_clone == pc_padding_value] = float('inf')
        # x_clone = torch.sum(x_clone, dim=-1)  # [B, N]
        # farthest = torch.argmin(x_clone, dim=1)  # [B]
        # farthest = farthest.long().to(device)
        """use a fixed point as the first point for fps to ensure order_invariant"""
        # Calculate distances from all points to the given point
        target_point = torch.tensor(
            [6.0622e-01, -4.4875e-05], device=xyz.device)[None, None, :]  # [1, 1, 2]
        distances__ = torch.norm(
            xyz[:, :, :2] - target_point, dim=-1)  # [B, N]
        # Find the index of the closest point
        if pc_padding_value is not None:
            distances__[pad_mask] = float('inf')
        farthest = torch.argmin(distances__, dim=1)  # [B]
        farthest = farthest.long().to(device)
    """
    above is to get the first farthest point index
     then iteratively get the remaining npoint-1 points
     """
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # [B, npoint]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)  # [B, 1, C]
        dist = torch.sum((xyz - centroid) ** 2, -1)  # [B, N]
        mask = dist < distance
        # the max distance, for the one have been sampled, the distance is 0
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz,
                     pc_padding_value: Optional[float] = None,
                     chunk_size: int = 256, order_invariant: bool = False):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, ndim], ndim=3 or 2
        new_xyz: query points, [B, S, ndim], S is n_points
        order_invariant:
            If True, the grouped indices (group_idx) are sorted by distance, ensuring that the output remains identical even if the input point cloud order is shuffled.
            If False, the grouping is not strictly order-invariant — results may vary slightly for shuffled inputs, but the differences are typically minor and negligible.
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    if not order_invariant:
        # use int32 to save memory, but need to convert to then.
        group_idx = torch.arange(N, dtype=torch.int32).to(
            device).view(1, 1, N).repeat([B, S, 1])  # [B, S, N]
        sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
        group_idx[sqrdists > radius**2] = N  # [B, S, N]
        if pc_padding_value is not None:
            # if the pad values are much larger than the radius^2
            # e.g. pad_value=10000,we no need this part
            pad_mask = xyz[:, :, 0] == pc_padding_value  # [B, N]
            pad_mask = pad_mask[:, None, :].repeat([1, S, 1])  # [B, S, N]
            group_idx[pad_mask] = N
        if N > 10000:
            # chunk the sort to avoid OOM
            sorted_parts = [
                group_idx[:, i: i + chunk_size, :].sort(dim=-1)[0]
                for i in range(0, S, chunk_size)
            ]
            group_idx = torch.cat(sorted_parts, dim=1)
        else:
            group_idx, _ = group_idx.sort(dim=-1)  # [B, S, N]

        group_idx = group_idx[:, :, :nsample]  # [B, S, nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat(
            [1, 1, nsample])  # [B, S, nsample], all first index
        mask = group_idx == N
        # if not enough samples, use the first one
        group_idx[mask] = group_first[mask]
    else:
        sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
        if pc_padding_value is not None:
            pad_mask = (xyz[:, :, 0] == pc_padding_value)  # [B, N]
            pad_mask = pad_mask[:, None,
                                :].expand(-1, sqrdists.size(1), -1)  # [B, S, N]
            sqrdists = sqrdists.masked_fill(
                pad_mask, 1e10)  # safer than 1e50
        # Mask points beyond the search radius
        # sqrdists = sqrdists.masked_fill(sqrdists > radius**2, float('inf'))
        # Sort distances to get nearest neighbors
        _, group_idx = sqrdists.sort(dim=-1)  # [B, S, N]
        group_idx = group_idx[:, :, ::2][:, :, :nsample]  # [B, S, nsample]
        # Identify invalid (masked) points
        # True where distance == inf
        # mask = torch.isinf(sqrdists[:, :, :nsample])
        # # Fallback to the first valid neighbor if not enough samples
        # # [B, S, nsample]
        # group_first = group_idx[:, :, 0:1].expand(-1, -1, nsample)
        # group_idx = torch.where(mask, group_first, group_idx)
    return group_idx.detach().cpu()


def real_sample_and_group_indices(
    npoint,
    radius,
    nsample,
    xyz,
    deterministic=False,
    fps_method: str = "first",
    pc_padding_value: Optional[float] = None,
    order_invariant: bool = False,
):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, ndim], ndim=3 or 2
        points: input points data, [B, N, D]
    Return:
        fps_idx: sampled points indices, [B, npoint] long format on device of xyz
        group_idx: group indices for each sample, [B,nsample, npoint], int32 format on cpu
    """
    B, N, C = xyz.shape
    S = npoint
    if fps_method == "fps":
        fps_idx = farthest_point_sample(
            xyz, npoint, pc_padding_value, deterministic=deterministic, order_invariant=order_invariant)  # [B, npoint, C]
    elif fps_method == "first":
        if pc_padding_value is None:
            fps_idx = torch.arange(npoint)[None].repeat(B, 1).to(xyz.device)
        else:
            # QB: if padding_value and the points is shuffled
            fps_idx = torch.arange(N).repeat(B, 1).to(xyz.device)
            mask = xyz[:, :, 0] != pc_padding_value
            # large_neg = torch.full_like(xyz[:, :, 0], float('inf'))
            fps_idx = torch.where(mask, fps_idx, float('inf'))
            fps_idx, _ = torch.sort(fps_idx, dim=1)
            fps_idx = fps_idx[:, :npoint].long()
    else:
        raise ValueError(f"Unknown FPS method: {fps_method}")
    # (B,npoint,C) take out the centroids points at fps_idx
    new_xyz = index_points(xyz, fps_idx)
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz,
                                 pc_padding_value=pc_padding_value, order_invariant=order_invariant)  # [B, npoint, nsample]

    return fps_idx, group_idx


def sample_and_group_indices(
    npoint,
    radius,
    nsample,
    xyz,
    deterministic=False,
    fps_method: str = "first",
    pc_padding_value: Optional[float] = None,
    sample_ids: Optional[torch.Tensor] = None,
    order_invariant: bool = False,
):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, ndim], ndim=3 or 2
        points: input points data, [B, N, D],
        order_invariant:
            If True, ensuring that the output remains identical even if the input point cloud order is shuffled.
            If False, results may vary slightly for shuffled inputs, but the differences are typically minor and negligible.

    Return:
        fps_idx: sampled points indices, [B, npoint] long format on device of xyz
        group_idx: group indices for each sample, [B,nsample, npoint], long format on device of xyz
    """
    if deterministic:
        if sample_ids is not None and \
                all(int(k) in CACHE_SAMPLE_AND_GROUP_INDECIES for k in sample_ids):
            fps_ids = []
            group_ids = []
            for k in sample_ids:
                fps_i, group_i = CACHE_SAMPLE_AND_GROUP_INDECIES[int(k)]
                fps_ids.append(fps_i)
                group_ids.append(group_i)

            return torch.stack(fps_ids), torch.stack(group_ids).to(xyz.device).long()
        else:
            fps_ids, group_ids = real_sample_and_group_indices(
                npoint, radius, nsample, xyz, deterministic, fps_method, pc_padding_value, order_invariant=order_invariant
            )
            if sample_ids is not None:
                for i, k in enumerate(sample_ids):
                    CACHE_SAMPLE_AND_GROUP_INDECIES[int(k)] = (
                        fps_ids[i], group_ids[i])
            return fps_ids, group_ids.to(xyz.device).long()
    else:
        fps_ids, group_ids = real_sample_and_group_indices(
            npoint, radius, nsample, xyz, deterministic, fps_method, pc_padding_value
        )
        return fps_ids, group_ids.long().to(xyz.device)


def sample_and_group(
    npoint,
    radius,
    nsample,
    xyz,
    points,
    returnfps=False,
    deterministic=False,
    fps_method: str = "first",
    pc_padding_value: Optional[float] = None,
    sample_ids: Optional[torch.Tensor] = None,
    order_invariant: bool = False,
):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, ndim], ndim=3 or 2
        points: input points data, [B, N, D]
        order_invariant:
            If True, the grouped indices (group_idx) are sorted by distance, ensuring that the output remains identical even if the input point cloud order is shuffled.
            If False, the grouping is not strictly order-invariant — results may vary slightly for shuffled inputs, but the differences are typically minor and negligible.
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, ndim]
        new_points: sampled points data, [B, npoint, nsample, ndim+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx, idx = sample_and_group_indices(
        npoint, radius, nsample, xyz, deterministic, fps_method, pc_padding_value, sample_ids=sample_ids, order_invariant=order_invariant
    )

    new_xyz = index_points(xyz, fps_idx)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D], C is 2 or 3, D is num of features (channels)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, ndim], ndim=3 or 2
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, ndim]
        new_points: sampled points data, [B, 1, N, ndim+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, deterministic=not self.training
            )
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
