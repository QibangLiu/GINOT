# %%
from typing import List, Optional, Tuple, Union
from modules.transformer import SelfAttentionBlocks, ResidualCrossAttentionBlock, MLP
from modules.point_encoding import PointSetEmbedding, SimplePerceiver
from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
from modules.sliding_window import SlidingWindow
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
def load_data(pv=-10):

    folder_path = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/dataset/pc_fieldoutput_11fram/dataset_0-10000_12-92"

    with open(os.path.join(folder_path, "mises_disp_laststep.pkl"), "rb") as f:
        mises_disp_data = pickle.load(f)
        SU_raw = mises_disp_data['mises_disp']
        su_shift = mises_disp_data['shift']
        su_scaler = mises_disp_data['scaler']

    with open(os.path.join(folder_path, "mesh_pc.pkl"), "rb") as f:
        data_mesh = pickle.load(f)
        coords = data_mesh['mesh_coords']
        connectivity = data_mesh['mesh_connect']
        point_cloud = data_mesh['point_cloud']
        sample_ids = data_mesh['valid_sample_ids']

    SU = [torch.tensor((su-su_shift)/su_scaler) for su in SU_raw]  # (Nb, N, 3)
    su_shift = torch.tensor(su_shift)[None, :]  # (1,1,3)
    su_scaler = torch.tensor(su_scaler)[None, :]  # (1, 1, 3)
    xyt = [torch.tensor(x[:, :2]) for x in coords]  # (Nb, N, 2)
    point_cloud = [torch.tensor(x[:, :2])
                   for x in point_cloud]  # (Nb,N,3)->(Nb, N, 2)
    SU = pad_sequence(SU, batch_first=True, padding_value=pv)
    xyt = pad_sequence(xyt, batch_first=True, padding_value=0)
    mask = (SU != pv).float()
    point_cloud = pad_sequence(point_cloud, batch_first=True, padding_value=0)
    IDs = np.arange(len(SU))
    train_pc, test_pc, train_xyt, test_xyt, train_su, test_su, train_mask, test_mask, train_idx, test_idx = train_test_split(
        point_cloud, xyt, SU, mask, IDs, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(
        train_pc, train_xyt, train_su, train_mask)
    test_dataset = TensorDataset(test_pc, test_xyt, test_su, test_mask)
    # data = {"train_dataset": train_dataset, "test_dataset": test_dataset, "su_shift": su_shift,
    #         "su_scaler": su_scaler, "SU_raw": SU_raw, "train_idx": train_idx, "test_idx": test_idx,
    #         "connectivity": connectivity, "sample_ids": sample_ids,
    #         }
    data = {"train_dataset": train_dataset, "test_dataset": test_dataset, "su_shift": su_shift,
            "su_scaler": su_scaler}
    return data


def plot_pc_node(pc, node):
    fig, ax = plt.subplots()
    ax.scatter(pc[:, 0], pc[:, 1], c='b', s=1)
    ax.scatter(node[:, 0], node[:, 1], c='r', s=1)
    plt.show()


# %%
data = load_data()
train_dataset = data["train_dataset"]
test_dataset = data["test_dataset"]
# %%
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
su_shift = data["su_shift"]
su_scaler = data["su_scaler"]


def match_dimensions(A, B):
    # Add singleton dimensions to B to match A's number of dimensions
    for _ in range(len(A.shape) - len(B.shape)):
        B = B.unsqueeze(0)
    return B


def mises_U_inverse(x, scaler: torch.tensor = su_scaler, shift: torch.tensor = su_shift):
    # scaler = match_dimensions(x, scaler)
    # shift = match_dimensions(x, shift)
    shift = shift.to(x.device)
    scaler = scaler.to(x.device)
    return x*scaler+shift


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
            final out shape: [B, latent_ctx, out_c]
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
            points (torch.Tensor): [B, N, C]
                   C =2 or 3, or >3 if has other features
        Returns:
            torch.Tensor: [B, out_c*latent_ctx]
        """
        # [B,N,C1]-->[B,C1,N]
        xyz = points.permute(0, 2, 1)

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
        # self.processor = SelfAttentionBlocks(width=embed_dim,
        #                              heads=4,
        #                              layers=4)
        self.output_proj = nn.Linear(
            embed_dim, out_channels)
        self.sliding_window = SlidingWindow(sliding_window_size)
        self.idx = 0

    def set_window_size(self, window_size):
        self.sliding_window.set_window(window_size)

    def sliding_fun(self, xyt, *args, **kwargs):
        branch_out = args[0]
        # B, N, T, C = xyt.shape
        # xyt = xyt.view(B, -1, C)
        x = self.Q_encoder(xyt)

        for block in self.resblocks:
            x = block(x, branch_out)
        # x = self.processor(x)

        x = self.output_proj(x)
        # x = x.view(B, N, T, -1)
        return x

    def forward(self, xyt, pc):
        self.idx += 1
        branch_out = self.branch(pc)
        # call self.sliding_fun(xyt, branch_out)
        # allocated_before = torch.cuda.memory_allocated() / 1e9
        # reserved_before = torch.cuda.memory_reserved() / 1e9
        x = self.sliding_window(xyt, self.sliding_fun, branch_out)
        # allocated_after = torch.cuda.memory_allocated() / 1e9
        # reserved_after = torch.cuda.memory_reserved() / 1e9
        # print(f"step{self.idx}, training{self.training}, allocated before: {allocated_before: .2f} GB, \
        #       allocated after: {allocated_after: .2f} GB, reserved before: {reserved_before: .2f} GB,\
        #       reserved after: {reserved_after: .2f} GB, diff_allocated: {allocated_after-allocated_before: .2f} GB,\
        #       diff_reserved: {reserved_after-reserved_before: .2f} GB")
        # x = self.Q_encoder(xyt)
        # for block in self.resblocks:
        #     x = block(x, branch_out)
        # x = self.processor(x)
        # x = self.output_proj(x)
        return x


# %%
embed_dim = 32
latent_ctx = 32
geo_encoder = PointCloudPerceiverChannelsEncoder(
    latent_ctx=latent_ctx, out_c=embed_dim)
pc = torch.randn(2, 484, 2).to(device)
geo_encoder.to(device)
h = geo_encoder(pc)
print(h.shape)
print("Total number of parameters of encoder: ", sum(p.numel()
      for p in geo_encoder.parameters()))

print("Allocated memory:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("Cached memory:", torch.cuda.memory_reserved() / 1024**3, "GB")
# %%
slidsize = 3000
model = Trunk(2, 3, embed_dim, geo_encoder, sliding_window_size=1000)
print("Total number of parameters of model: ", sum(p.numel()
      for p in model.parameters()))
model = model.to(device)
# x_in = torch.randn(2, 1000, 2).to(device)

# out = model(x_in, pc.to(device))
# %%


class TorchTrainer(torch_trainer.TorchTrainer):
    def __init__(self, model, device, filebase):
        super().__init__(model, device, filebase)

    def evaluate_losses(self, data):
        pc, xyt, mises_disp, mask = [d.to(self.device) for d in data]
        y_pred = self.models[0](xyt, pc)
        loss = nn.MSELoss(reduction='none')(y_pred, mises_disp)
        # TODO: mask.sum() may be zero, for sliding window, so add 1 to avoid division by zero
        loss = (loss * mask).sum() / (mask.sum()+1)
        loss_dic = {"loss": loss.item()}
        return loss, loss_dic

    def predict(self, data_loader):
        y_pred = []
        y_true = []
        self.models[0].eval()
        with torch.no_grad():
            for data in data_loader:
                pc, xyt, mises_disp, mask = [d.to(self.device) for d in data]
                mask = mask.bool()
                pred = self.models[0](xyt, pc)
                pred = mises_U_inverse(pred)
                y_true_inv = mises_U_inverse(mises_disp)
                pred = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                        for x, i in zip(pred, mask)]
                y_true_inv = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                              for x, i in zip(y_true_inv, mask)]

                y_pred = y_pred+pred
                y_true = y_true+y_true_inv

        return y_pred, y_true


# %%
parser = argparse.ArgumentParser(
    description="Model training arguments")
parser.add_argument(
    "--train_flag", type=str, default="start")
parser.add_argument("--epochs", type=int, default=1)
args, unknown = parser.parse_known_args()

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

checkpoint = torch_trainer.ModelCheckpoint(
    monitor="val_loss", save_best_only=True)
file_base = f"./saved_models/last_step_lctx{latent_ctx}_embed{embed_dim}"
print("file_base:", file_base)
trainer = TorchTrainer(model, device, file_base)
optimizer = torch.optim.Adam(trainer.parameters(), lr=5e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=10)


trainer.compile(
    optimizer=optimizer,
    checkpoint=checkpoint,
    loss_fn=nn.MSELoss(),
    lr_scheduler=lr_scheduler,
    window_size=slidsize,
    sequence_idx=[1, 2, 3]
)

if not args.train_flag == "start":
    trainer.load_weights(device=device)
    h = trainer.load_logs()
if args.train_flag == "continue" or args.train_flag == "start":
    h = trainer.fit(
        train_loader, val_loader=test_loader, epochs=args.epochs
    )
    trainer.save_logs()
    trainer.load_weights(device=device)

if h is not None:
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    if "loss" in h:
        ax.plot(h["loss"], label="loss")
    if "val_loss" in h:
        ax.plot(h["val_loss"], label="val_loss")
    ax.legend()
    ax.set_yscale("log")

# %%
y_pred, y_true = trainer.predict(test_loader)

# %%
error_s = []
for y_p, y_t in zip(y_pred, y_true):
    s_p, s_t = y_p[:, 0], y_t[:, 0]
    ux_p, ux_t = y_p[:, 1], y_t[:, 1]
    uy_p, uy_t = y_p[:, 2], y_t[:, 2]
    e_s = np.linalg.norm(s_p-s_t)/np.linalg.norm(s_t)
    e_ux = np.linalg.norm(ux_p-ux_t)/np.linalg.norm(ux_t)
    e_uy = np.linalg.norm(uy_p-uy_t)/np.linalg.norm(uy_t)

    error_s.append((e_s+e_ux+e_uy)/3)

error_s = np.array(error_s)

fig = plt.figure()
_ = plt.hist(error_s)
plt.xlabel("L2 relative error")
plt.ylabel("frequency")

mu, std = np.mean(error_s), np.std(error_s)
print("Mean of L2 relative error:", mu)
print("Standard deviation of L2 relative error:", std)

# %%
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[-1]
median_index = sort_idx[len(sort_idx) // 2]

# Print the indexes
print("Index for minimum element:", min_index, "Error:", error_s[min_index])
print("Index for maximum element:", max_index, "Error:", error_s[max_index])
print("Index for median element:", median_index,
      "Error:", error_s[median_index])

# %%
min_median_max_index = np.array([min_index, median_index, max_index])


nr, nc = 2, 3
fig = plt.figure(figsize=(nc*4.8, nr*3.6))
for i, index in enumerate(min_median_max_index):
    coord = test_dataset[index][1].cpu().detach().numpy()
    p_true = y_true[index][:, 0]
    coord = coord[:p_true.shape[0]]
    p_pred = y_pred[index][:, 0]

    ax = plt.subplot(nr, nc, i+1)
    c_t = ax.scatter(coord[:, 0], coord[:, 1], c=p_true,
                     cmap='viridis', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mises Stress (True)')
    cbar = fig.colorbar(c_t, ax=ax, label='Mises Stress')
    plt.tight_layout()

    ax = plt.subplot(nr, nc, i+4)
    cp = ax.scatter(coord[:, 0], coord[:, 1], c=p_pred,
                    cmap='viridis', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mises Stress (Pred.)')
    cbar = fig.colorbar(c_t, ax=ax, label='Mises Stress')
    plt.tight_layout()
