# %%

from typing import List, Optional, Tuple, Union
from modules.transformer import SelfAttentionBlocks, ResidualCrossAttentionBlock, MLP
from modules.point_encoding import PointSetEmbedding, SimplePerceiver
from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import timeit
import os
import pickle
from sklearn.model_selection import train_test_split
import trainer.torch_trainer as torch_trainer
from skimage import measure
from shapely.geometry import Polygon
import argparse
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# def load_data(sets=[0]):

#     file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/GeoSDF2D/dataset/pc_fieldoutput/"

#     mises_all = []
#     disp_all = []
#     coords_all = []
#     connectivity_all = []
#     point_cloud_all = []
#     sample_ids_all = []
#     for setID in sets:
#         folder_path = os.path.join(file_base, "dataset_"+str(setID))
#         with open(os.path.join(folder_path, "mises.pkl"), "rb") as f:
#             mises = pickle.load(f)['mises']
#         with open(os.path.join(folder_path, "disp.pkl"), "rb") as f:
#             disp = pickle.load(f)['disp']
#         with open(os.path.join(folder_path, "mesh_pc.pkl"), "rb") as f:
#             data_mesh = pickle.load(f)
#             coords = data_mesh['mesh_coords']
#             connectivity = data_mesh['mesh_connect']
#             point_cloud = data_mesh['point_cloud']
#             sample_ids = data_mesh['valid_sample_ids']

#         mises_all = mises_all+mises
#         disp_all = disp_all+disp
#         coords_all = coords_all+coords
#         connectivity_all = connectivity_all+connectivity
#         point_cloud_all.append(point_cloud[:, :, :2])
#         sample_ids_all.append(sample_ids)
#     point_cloud_all = np.concatenate(point_cloud_all)
#     sample_ids_all = np.concatenate(sample_ids_all)
#     numcase = 100
#     return mises_all[:numcase], disp_all[:numcase], coords_all[:numcase], point_cloud_all[:numcase], connectivity_all[:numcase], sample_ids_all[:numcase]


# def plot_pc_node(pc, node):
#     fig, ax = plt.subplots()
#     ax.scatter(pc[:, 0], pc[:, 1], c='b', s=1)
#     ax.scatter(node[:, 0], node[:, 1], c='r', s=1)
#     plt.show()


# # %%
# mises_raw, disp_raw, coords, point_cloud, _, _ = load_data([0])
# # mises_raw: (Nb, Nt,N), N is varing for different samples
# # disp_raw: (Nb, Nt,N, 3)
# data_concat = np.concatenate([x.flatten() for x in mises_raw])
# mises_shift = np.mean(data_concat)
# mises_scaler = np.std(data_concat)
# del data_concat  # Release memory

# data_concat = np.concatenate(
#     [x.reshape(-1, disp_raw[0].shape[-1]) for x in disp_raw])
# disp_shift = np.mean(data_concat, axis=0)
# disp_scaler = np.std(data_concat, axis=0)
# del data_concat  # Release memory

# mises_disp_shift = torch.tensor(
#     np.concatenate((np.array([mises_shift]), disp_shift)))
# mises_disp_scaler = torch.tensor(
#     np.concatenate((np.array([mises_scaler]), disp_scaler)))
# mises_disp_shift = mises_disp_shift[None, None, :]
# mises_disp_scaler = mises_disp_scaler[None, None, :]
# Nt = mises_raw[0].shape[0]
# t = torch.tensor(np.linspace(0, 1, Nt, dtype=np.float32))
# t = t[None, :, None]
# # %%
# # mises = [torch.tensor((x-mises_shift)/mises_scaler) for x in mises_raw]
# # i = 0
# # disp0 = [torch.tensor((x[:, i]-disp_shift[i])/disp_scaler[i])
# #          for x in disp_raw]
# # i = 1
# # disp1 = [torch.tensor((x[:, i]-disp_shift[i])/disp_scaler[i])
# #          for x in disp_raw]

# mises_disp = [(((torch.tensor(np.concatenate((s[:, :, None], d), axis=-1))-mises_disp_shift)/mises_disp_scaler)).permute(1, 0, 2)
#               for s, d in zip(mises_raw, disp_raw)]  # [(Nb N, Nt, 3)]


# xyt = [torch.cat((torch.tensor(x[:, None, :2]).repeat(
#     1, Nt, 1), t.repeat(x.shape[0], 1, 1)), dim=-1) for x in coords]  # [(Nb, N, Nt, 3)]

# point_cloud = torch.tensor(point_cloud).permute(0, 2, 1)  # [(Nb, 2, N)]

# mises_disp = pad_sequence(mises_disp, batch_first=True, padding_value=0)
# xyt = pad_sequence(xyt, batch_first=True, padding_value=0)
# mask = (mises_disp != 0).float()


# # %%
# # Split the data into training and testing sets
# train_pc, test_pc, train_xyt, test_xyt, train_mises_disp, test_mises_disp, train_mask, test_mask = train_test_split(
#     point_cloud, xyt, mises_disp, mask, test_size=0.2, random_state=42)

# train_dataset = TensorDataset(
#     train_pc, train_xyt, train_mises_disp, train_mask)
# test_dataset = TensorDataset(test_pc, test_xyt, test_mises_disp, test_mask)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%


class PointCloudPerceiverChannelsEncoder(nn.Module):
    """
    Encode point clouds using a transformer model with an extra output
    token used to extract a latent vector.
    """

    def __init__(self,
                 input_channels: int = 2,
                 out_c: int = 128,
                 width: int = 128,
                 latent_ctx: int = 128,
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
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Args:
            input_channels (int): 2 or 3
            width (int): hidden dimension
            latent_ctx (int): number of context points
            n_point (int): number of points in the point set embedding
            n_sample (int): number of samples in the point set embedding
            radius (float): radius for the point set embedding
            patch_size//2 (int): padding size of dim 1 of conv in the point set embedding
            padding_mode (str): padding mode of the conv in the point set embedding
            d_hidden (list): hidden dimensions for the conv in the point set embedding
            fps_method (str): method for point sampling in the point set embedding, 'fps' or 'first', 'fps' has issue
            out_c (int): output channels
            final out shape: [B, out_c*latent_ctx]
        """
        self.width = width
        self.latent_ctx = latent_ctx
        self.n_point = n_point
        self.out_c = out_c
        # position embeding + linear layer
        self.pos_emb_linear = PosEmbLinear("nerf", input_channels, self.width)

        d_input = self.width
        self.point_set_embedding \
            = PointSetEmbedding(ndim=input_channels, radius=radius, n_point=self.n_point,
                                n_sample=n_sample, d_input=d_input,
                                d_hidden=d_hidden, patch_size=patch_size,
                                padding_mode=padding_mode,
                                fps_method=fps_method)

        self.register_parameter(
            "output_tokens",
            nn.Parameter(torch.randn(self.latent_ctx, self.width)),
        )
        self.ln_pre = nn.LayerNorm(self.width)
        self.ln_post = nn.LayerNorm(self.width)

        self.encoder = SimplePerceiver(
            width=self.width, heads=num_heads, layers=cross_attn_layers)

        self.processor = SelfAttentionBlocks(
            width=self.width, heads=num_heads, layers=self_attn_layers)
        self.output_proj = nn.Linear(
            self.width, self.out_c)

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): [B, C, N]
                   C =2 or 3, or >3 if has other features
        Returns:
            torch.Tensor: [B, out_c*latent_ctx]
        """
        xyz = points
        # [B,C1,N] -> [B,N,C1]
        points = points.permute(0, 2, 1)
        # [B, N, C1] -> [B, N, C2], C2=self.width
        dataset_emb = self.pos_emb_linear(points)  # [B, N, C]
        # [B, N, C2] -> [B, C2, N]
        points = dataset_emb.permute(0, 2, 1)
        # [B, C2, N] -------------> [B, C3, No], No=n_point
        #      \ pointNet             / mean (dim=2)
        #       \ permute            / Conv, C3=d_hidden[-1]
        #       [B, C2+ndim,  n_sample, n_point]
        data_tokens = self.point_set_embedding(xyz, points)
        # [B, Co, No] -> [B, No, Co]
        data_tokens = data_tokens.permute(0, 2, 1)
        batch_size = points.shape[0]
        latent_tokens = self.output_tokens.unsqueeze(
            0).repeat(batch_size, 1, 1)  # [B, latent_ctx, width]
        # [B, n_point+latent_ctx, width]
        h = self.ln_pre(torch.cat([data_tokens, latent_tokens], dim=1))
        assert h.shape == (batch_size, self.n_point +
                           self.latent_ctx, self.width)
        # [B, n_point+latent_ctx, width] -> [B,  n_point+latent_ctx, width]
        h = self.encoder(h, dataset_emb)
        h = self.processor(h)
        # [B,  n_point+latent_ctx, width] -> [B, latent_ctx, width]
        # -> [B, latent_ctx, out_c]
        h = self.output_proj(self.ln_post(h[:, -self.latent_ctx:]))
        return h


class SlidingWindow():
    def __init__(self, window_size, overlap=0):
        self.window_size = window_size
        self.stride = window_size - overlap

    def set_window(self, window_size, overlap=0):
        self.window_size = window_size
        self.stride = window_size - overlap

    def __call__(self, sequence, fun: callable, *args, **kwargs):
        sequence_length = sequence.size(1)
        outputs = []
        for start in range(0, sequence_length, self.stride):
            end = min(start + self.window_size, sequence_length)
            window = sequence[:, start:end]
            output = fun(window, *args, **kwargs)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Trunk(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, branch, sliding_window_size=400):
        super().__init__()
        self.Q_encoder = MLP(embed_dim, in_channels)
        self.branch = branch
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=embed_dim,
                    heads=4
                )
                for _ in range(4)
            ]
        )
        self.processor = SelfAttentionBlocks(width=embed_dim,
                                     heads=4,
                                     layers=4)
        self.output_proj = nn.Linear(
            embed_dim, out_channels)
        self.sliding_window = SlidingWindow(sliding_window_size)

    def set_window_size(self, window_size):
        self.sliding_window.set_window(window_size)

    def sliding_fun(self, xyt, *args, **kwargs):
        branch_out = args[0]
        B, N, T, C = xyt.shape
        xyt = xyt.view(B, -1, C)
        x = self.Q_encoder(xyt)
        for block in self.resblocks:
            x = block(x, branch_out)
        x = self.processor(x)
        x = self.output_proj(x)
        x = x.view(B, N, T, -1)
        return x

    def forward(self, xyt, pc):
        branch_out = self.branch(pc)
        # call self.sliding_fun(xyt, branch_out)
        x = self.sliding_window(xyt, self.sliding_fun, branch_out)
        # x = self.Q_encoder(xyt)
        # for block in self.resblocks:
        #     x = block(x, branch_out)
        # x = self.processor(x)
        # x = self.output_proj(x)
        return x


# %%
embed_dim = 64
geo_encoder = PointCloudPerceiverChannelsEncoder(
    latent_ctx=64, out_c=embed_dim)
pc = torch.randn(2, 2, 484).to(device)
geo_encoder.to(device)
h = geo_encoder(pc)
print(h.shape)
print("Total number of parameters of encoder: ", sum(p.numel()
      for p in geo_encoder.parameters()))

print("Allocated memory:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("Cached memory:", torch.cuda.memory_reserved() / 1024**3, "GB")
# %%
model = Trunk(3, 3, embed_dim, geo_encoder, sliding_window_size=50)
print("Total number of parameters of model: ", sum(p.numel()
      for p in model.parameters()))
model = model.to(device)
x_in = torch.randn(2, 8000, 26, 3).to(device)

out = model(x_in, pc.to(device))
# %%


class TorchTrainer(torch_trainer.TorchTrainer):
    def __init__(self, model, device, filebase):
        super().__init__(model, device, filebase)

    def evaluate_losses(self, data):
        pc, xyt, mises_disp, mask = [d.to(self.device) for d in data]
        y_pred = self.models[0](xyt, pc)
        loss = nn.MSELoss(reduction='none')(y_pred, mises_disp)
        loss = (loss * mask).sum() / mask.sum()
        loss_dic = {"loss": loss.item()}
        return loss, loss_dic

    # def predict(self, data_loader):
    #     y_pred = []
    #     y_true = []
    #     self.models[0].eval()
    #     with torch.no_grad():
    #         for data in data_loader:
    #             x_branch = data[0].to(self.device)
    #             pred = self.models[0](x_trunk, x_branch)
    #             pred = pred.cpu().detach().numpy()
    #             y_pred.append(pred)
    #             y_true.append(data[1].cpu().detach().numpy())
    #     y_true = np.vstack(y_true)
    #     y_pred = np.vstack(y_pred)
    #     return y_pred, y_true
