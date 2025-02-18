# %%
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
current_work_path = os.getcwd()
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_work_path == current_file_dir:
    from configs import models_configs, LoadData
    from geoencoder import LoadGeoEncoderModel, GeoEncoderModelDefinition
    from modules.UNets import UNet
    from trainer import torch_trainer
    from modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
else:
    from .configs import models_configs, LoadData
    from .geoencoder import LoadGeoEncoderModel, GeoEncoderModelDefinition
    from .modules.UNets import UNet
    from .trainer import torch_trainer
    from .modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%


class Branch(nn.Module):
    def __init__(
        self,
        geo_encoder,
        sdf_NN,
        embed_dim=64,
        img_shape=(1, 120, 120),
        first_conv_channels=8,
        channel_mutipliers=[1, 2, 4, 8],
        has_attention=[False, False, True, True],
        num_res_blocks=1,
        norm_groups=8,
        dropout=None
    ):
        super().__init__()
        self.geo_encoder = geo_encoder
        self.sdf_NN = sdf_NN
        for param in geo_encoder.parameters():
            param.requires_grad = False
        self.geo_encoder.eval()
        for param in sdf_NN.parameters():
            param.requires_grad = False
        self.sdf_NN.eval()

        self.unet = UNet(
            img_shape,
            first_conv_channels,
            channel_mutipliers,
            has_attention,
            num_res_blocks=num_res_blocks,
            norm_groups=norm_groups,
            dropout=dropout
        )
        in_channels = channel_mutipliers[0] * first_conv_channels
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, stride=2, padding=1))
        self.convs.append(nn.SiLU())
        self.convs.append(nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1))
        self.convs.append(nn.SiLU())
        self.convs.append(nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1))
        self.convs.append(nn.SiLU())
        self.convs.append(nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=1, padding=1))
        self.convs.append(nn.SiLU())

        self.transformer = SelfAttentionBlocks(
            width=embed_dim,
            heads=4,
            layers=2,
        )

    def forward_from_sdf(self, normalized_sdf):
        # (B, 1, 120, 120)->(B, Cin, 120, 120)
        x = self.unet(normalized_sdf)
        # (B, Cin, 120, 120)->(B, embed_dim, 30, 30)
        for layer in self.convs:
            x = layer(x)
        # (B, embed_dim, 30, 30)->(B,embed_dim, 30*30)
        x = x.view(x.size(0), x.size(1), -1)
        # (B,embed_dim, 30*30)->(B, 30*30, embed_dim)
        x = x.permute(0, 2, 1)
        # (B, 30*30, embed_dim)->(B, 30*30, embed_dim)
        x = self.transformer(x)
        return x

    def forward(self, pc, grid_points):
        """
        Args:
            pc: point cloud (B,N,2)
            grid_points: grid points (Nx*Ny,2) Nx=Ny=120
        """
        # (B, N, 2)->(B,latent_dim,out_c)
        latent = self.geo_encoder(pc, apply_padding_pointnet2=True)
        normalized_sdf = self.sdf_NN(grid_points, latent)  # (B, N)
        normalized_sdf = normalized_sdf.view(-1, 1, 120, 120)
        x = self.forward_from_sdf(normalized_sdf)
        return x


class Trunk(nn.Module):
    def __init__(self, branch, embed_dim=64, padding_value=-10, in_channels=2, out_channels=3, emd_version="nerf"):
        super().__init__()
        self.padding_value = padding_value
        # d = position_encoding_channels(emd_version)
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
        self.output_proj = nn.Linear(
            embed_dim, out_channels)

    def forward(self, xyt, pc, grid_points):
        sdf = self.branch(pc, grid_points)  # (B, 60*60, embed_dim)
        # (B,N,2)->(B,N,embed_dim)
        # xyt = encode_position('nerf', position=xyt)
        x = self.Q_encoder(xyt)
        for block in self.resblocks:
            x = block(x, sdf)  # (B, N, embed_dim)
        # (B, N, embed_dim)->(B, N, 3)
        x = self.output_proj(x)
        return x.squeeze(-1)


def NTOModelDefinition(geo_encoder, sdf_NN, embed_dim=64, img_shape=(1, 120, 120),
                       channel_mutipliers=[1, 2, 4, 8],
                       has_attention=[False, False, True, True],
                       first_conv_channels=8, num_res_blocks=1,
                       norm_groups=8, dropout=None, padding_value=-10):
    branch = Branch(geo_encoder,
                    sdf_NN,
                    embed_dim,
                    img_shape,
                    first_conv_channels,
                    channel_mutipliers,
                    has_attention,
                    num_res_blocks,
                    norm_groups,
                    dropout)
    trunk = Trunk(branch, embed_dim, padding_value)
    tot_num_params = sum(p.numel() for p in trunk.parameters())
    trainable_params = sum(p.numel()
                           for p in trunk.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of NTO model: {tot_num_params}, {trainable_params} of which are trainable")

    return trunk


# %%


def LoadNTOModel(filebase, NTO_model_args, geo_encoder_model_args):
    model_path = os.path.join(filebase, "model.ckpt")
    geo_encoder, sdf_NN = GeoEncoderModelDefinition(geo_encoder_model_args)
    NTO_model = NTOModelDefinition(geo_encoder, sdf_NN, **NTO_model_args)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    NTO_model.load_state_dict(state_dict)
    NTO_model.to(device)
    NTO_model.eval()
    return NTO_model


def TrainNTOModel(NTO_model, filebase, train_flag, epochs=300, lr=1e-3):

    train_dataset, test_dataset, grid_coors, su_inverse = LoadData(
        seed=42)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    grid_coors = grid_coors.to(device)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            pc = data[0].to(self.device)
            xyt = data[1].to(self.device)
            y_true = data[2].to(self.device)
            mask = (y_true != self.models[0].padding_value).float()
            y_pred = self.models[0](xyt, pc, grid_coors)
            loss = nn.MSELoss(reduction='none')(y_true, y_pred)
            loss = (loss*mask).sum()/mask.sum()
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data_loader, grid_coors):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data_loader:
                    pc = data[0].to(self.device)
                    xyt = data[1].to(self.device)
                    y_true_batch = data[2].to(self.device)
                    mask = (y_true_batch != self.models[0].padding_value)
                    pred = self.models[0](xyt, pc, grid_coors)
                    pred = su_inverse(pred)
                    y_true_batch = su_inverse(y_true_batch)

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
    )
    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(train_loader, val_loader=test_loader,
                    epochs=epochs, print_freq=1)
    trainer.save_logs()

    EvaluateForwardModel(trainer, test_loader, grid_coors)
    return trainer


def EvaluateForwardModel(trainer, test_loader, grid_coors):
    trainer.load_weights(device=device)
    y_pred, y_true = trainer.predict(test_loader, grid_coors)

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

    sort_idx = np.argsort(error_s)
    idx_best = sort_idx[0]
    idx_32perc = sort_idx[int(len(sort_idx)*0.32)]
    idx_63perc = sort_idx[int(len(sort_idx)*0.63)]
    idx_95perc = sort_idx[int(len(sort_idx)*0.95)]
    index_list = [idx_best, idx_32perc, idx_63perc, idx_95perc]
    labels = ["Best", "32th percentile", "63th percentile", "95th percentile"]
    for label, idx in zip(labels, index_list):
        print(f"{label} L2 error: {error_s[idx]}")

    print(f"Mean L2 error: {np.mean(error_s)}, std: {np.std(error_s)}")


# %%
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training arguments")
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    configs = models_configs()

    filebase = configs["NTOModel"]["filebase"]
    model_args = configs["NTOModel"]["model_args"]
    geo_encoder_filebase = configs["GeoEncoder"]["filebase"]
    geo_encoder_model_args = configs["GeoEncoder"]["model_args"]
    print(f"\n\nForwardModel Filebase: {filebase}, model_args:")
    print(model_args)
    print(f"\n\nGeoEncoder Filebase: {geo_encoder_filebase}, model_args:")
    print(geo_encoder_model_args)

    geo_encoder, sdf_NN = LoadGeoEncoderModel(
        geo_encoder_filebase, geo_encoder_model_args)
    NTO_model = NTOModelDefinition(geo_encoder, sdf_NN, **model_args)

    trainer = TrainNTOModel(NTO_model, filebase, args.train_flag,
                            epochs=args.epochs, lr=args.learning_rate)
    print(filebase, " training finished")

# %%
