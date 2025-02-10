# %%
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
if __package__:
    from . import configs
    from .geoencoder import LoadGeoEncoderModel, GeoEncoderModelDefinition
    from .modules.UNets import UNet
    from .trainer import torch_trainer
    from .modules.transformer import Transformer, MLP, ResidualCrossAttentionBlock
    from .modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
else:
    import configs
    from geoencoder import LoadGeoEncoderModel, GeoEncoderModelDefinition
    from modules.UNets import UNet
    from trainer import torch_trainer
    from modules.transformer import Transformer, MLP, ResidualCrossAttentionBlock
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%


class Branch(nn.Module):
    def __init__(
        self,
        geo_encoder,

    ):
        super().__init__()
        self.geo_encoder = geo_encoder

        latent_d = geo_encoder.latent_d
        out_c = geo_encoder.out_c

    def forward(self, pc):
        """
        Args:
            pc: point cloud (B,N,2)
            grid_points: grid points (Nx*Ny,2) Nx=Ny=120
        """
        # (B, N, 2)->(B,latent_dim,out_c)
        x = self.geo_encoder(pc)

        return x


# %%

class Trunk(nn.Module):
    def __init__(self, branch, embed_dim=64, cross_attn_layers=4, num_heads=4,
                 in_channels=3, out_channels=1,
                 dropout=0.0, emd_version="nerf", padding_value=-10):
        super().__init__()
        self.padding_value = padding_value
        d = position_encoding_channels(emd_version)
        # self.Q_encoder = MLP(embed_dim, in_channels)
        self.Q_encoder = nn.Sequential(nn.Linear(d*in_channels, 2*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(2*embed_dim, 3*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(3*embed_dim, 3*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(3*embed_dim, 3*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(3*embed_dim, 2*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(2*embed_dim, embed_dim)
                                       )
        self.branch = branch
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=embed_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(cross_attn_layers)
            ]
        )
        self.output_proj = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(2*embed_dim, 2*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(2*embed_dim, out_channels)
                                         )
        self.step = 0

    def forward(self, xyt, pc):
        self.step += 1
        print(f"Step {self.step}, training{self.training}")
        print(
            f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(
            f"Cached memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        latent = self.branch(pc)  # (B, latenc, embed_dim)
        print(
            f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(
            f"Cached memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        # (B,N,ndim)->(B,N,embed_dim)
        xyt = encode_position('nerf', position=xyt)
        x = self.Q_encoder(xyt)
        for block in self.resblocks:
            x = block(x, latent)  # (B, N, embed_dim)
        # (B, N, embed_dim)->(B, N, 1)
        x = self.output_proj(x)
        return x.squeeze(-1)


# %%

def NTOModelDefinition(branch_args, trunc_args):
    geo_encoder, _ = GeoEncoderModelDefinition(**branch_args)
    branch = Branch(geo_encoder)
    trunk = Trunk(branch, **trunc_args)
    tot_num_params = sum(p.numel() for p in trunk.parameters())
    trainable_params = sum(p.numel()
                           for p in trunk.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of NTO model: {tot_num_params}, {trainable_params} of which are trainable")

    return trunk


# %%
def EvaluateForwardModel(trainer, test_loader, train_loader):
    trainer.load_weights(device=device)

    def cal_l2_error(test_loader):
        y_pred, y_true = trainer.predict(test_loader)
        error_s = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_s = np.linalg.norm(s_p-s_t)/np.linalg.norm(s_t)
            error_s.append(e_s)
        error_s = np.array(error_s)
        return error_s

    error_s = cal_l2_error(test_loader)
    sort_idx = np.argsort(error_s)
    idx_best = sort_idx[0]
    idx_32perc = sort_idx[int(len(sort_idx)*0.32)]
    idx_63perc = sort_idx[int(len(sort_idx)*0.63)]
    idx_99perc = sort_idx[int(len(sort_idx)*0.99)]
    index_list = [idx_best, idx_32perc, idx_63perc, idx_99perc]
    labels = ["Best", "32th percentile", "63th percentile", "99th percentile"]
    for label, idx in zip(labels, index_list):
        print(f"{label} L2 error: {error_s[idx]}")

    print(
        f"Mean L2 error for test data: {np.mean(error_s)}, std: {np.std(error_s)}")

    error_s = cal_l2_error(train_loader)
    print(
        f"Mean L2 error for training data: {np.mean(error_s)}, std: {np.std(error_s)}")


def TrainNTOModel(NTO_model, filebase, train_flag, epochs=300, lr=1e-3):

    train_dataset, test_dataset, cells_train, cells_test, s_inverse, pc_inverse, vert_inverse = configs.LoadDataGEJEBGeo()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            pc = data[0].to(self.device)
            xyt = data[1].to(self.device)
            y_true = data[2].to(self.device)
            mask = (y_true != self.models[0].padding_value).float()
            y_pred = self.models[0](xyt, pc)
            loss = nn.MSELoss(reduction='none')(y_true, y_pred)
            loss = (loss*mask).sum()/(mask.sum()+1)
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data_loader):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data_loader:
                    pc = data[0].to(self.device)
                    xyt = data[1].to(self.device)
                    y_true_batch = data[2].to(self.device)
                    mask = (y_true_batch != self.models[0].padding_value)
                    pred = self.models[0](xyt, pc)
                    pred = s_inverse(pred)
                    y_true_batch = s_inverse(y_true_batch)
                    pred = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                            for x, i in zip(pred, mask)]
                    y_true_batch = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                                    for x, i in zip(y_true_batch, mask)]

                    y_pred = y_pred+pred
                    y_true = y_true+y_true_batch

            return y_pred, y_true

    trainer = TRAINER(
        NTO_model, device, filebase)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="val_loss", save_best_only=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=40)
    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="val_loss",
        window_size=600,
        sequence_idx=[1, 2],
    )
    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(train_loader, val_loader=test_loader,
                    epochs=epochs, print_freq=1)
    trainer.save_logs()

    EvaluateForwardModel(trainer, test_loader, train_loader)
    return trainer


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    configs_geo_from_pc = configs.JEB_geo_from_pc_configs()

    filebase = configs_geo_from_pc["filebase"]
    trunk_args = configs_geo_from_pc["trunk_args"]
    branch_args = configs_geo_from_pc["branch_args"]
    print(configs_geo_from_pc)

    NTO_model = NTOModelDefinition(branch_args, trunk_args)

    trainer = TrainNTOModel(NTO_model, filebase, args.train_flag,
                            epochs=args.epochs, lr=args.learning_rate)
    print(filebase, " training finished")

# %%
