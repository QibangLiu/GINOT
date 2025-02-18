# %%
from typing import List, Optional, Tuple, Union
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
if __package__:
    from .configs import models_configs, LoadData
    from .modules.params_proj import ChannelsParamsProj
    from .modules.transformer import SelfAttentionBlocks
    from .modules.point_encoding import PointSetEmbedding, SimplePerceiver
    from .modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from .trainer import torch_trainer
else:
    from configs import models_configs, LoadData
    from modules.params_proj import ChannelsParamsProj
    from modules.transformer import SelfAttentionBlocks
    from modules.point_encoding import PointSetEmbedding, SimplePerceiver
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
    from trainer import torch_trainer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%


class implicit_sdf(nn.Module):
    def __init__(self, latent_d=64, ndim=2, d_hidden_sdfnn=[256, 128], emd_version="nerf"):

        super().__init__()
        self.params_pre_name = 'projed_mlp_'
        self.emd_version = emd_version
        d = position_encoding_channels(emd_version)
        weight_shapes = [torch.Size([latent_d//4, d*ndim]), torch.Size(
            [latent_d//4, latent_d//4]),  torch.Size([latent_d//4, latent_d//4]), torch.Size([latent_d//4, latent_d//4])]

        self.param_shapes = {}

        for i, v in enumerate(weight_shapes):
            self.param_shapes[self.params_pre_name+str(i)+'_weight'] = v
            self.register_parameter(
                self.params_pre_name+str(i)+'_bias', nn.Parameter(torch.randn(v[0])))

        self.latent_d = latent_d
        learned_scale = 0.0625
        use_ln = True
        self.params_proj = ChannelsParamsProj(
            device=device,
            param_shapes=self.param_shapes,
            d_latent=self.latent_d,
            learned_scale=learned_scale,
            use_ln=use_ln,
        )
        self.nn_layers = nn.ModuleList()
        c_i = latent_d//4
        for ls in d_hidden_sdfnn:
            self.nn_layers.append(nn.Linear(c_i, ls))
            self.nn_layers.append(nn.SiLU())
            c_i = ls
        self.nn_layers.append(nn.Linear(c_i, 1))

    def forward(self, x, latent):
        """
        Args:
            x (torch.Tensor): [N, 2], query points,
            latent (torch.Tensor): [B, latent_d, C], latent vectors
        """
        # (N,2)--> (N,62)
        x = encode_position('nerf', position=x)
        # (1,N,62)->(B,N,62)
        x = x[None].repeat(latent.shape[0], 1, 1)
        # latent = latent.view(latent.shape[0], self.d_latent, -1)
        proj_params = self.params_proj(latent)
        for i, kw in enumerate(proj_params.keys()):
            w = proj_params[kw]
            b = getattr(self, self.params_pre_name+str(i)+'_bias')
            # (B,P,I)-> (B,P,O)
            x = torch.einsum("bpi,boi->bpo", x, w)
            x = torch.add(x, b)
            x = F.silu(x)

        for layer in self.nn_layers:
            x = layer(x)
        # (B,N,1)->(B,N)
        return x.squeeze()


def GeoEncoderModelDefinition(input_channels=2, out_c=128, latent_d=128,
                              width=128, n_point=128,
                              n_sample=8, radius=0.2,
                              d_hidden=[128, 128],
                              fps_method: str = 'first',
                              num_heads=4, cross_attn_layers=1,
                              self_attn_layers=3,
                              pc_padding_val: Optional[int] = None,
                              dropout=0.0):

    geo_encoder = PointCloudPerceiverChannelsEncoder(
        input_channels=input_channels, out_c=out_c, width=width,
        latent_d=latent_d, n_point=n_point, n_sample=n_sample,
        radius=radius, d_hidden=d_hidden, fps_method=fps_method,
        num_heads=num_heads, cross_attn_layers=cross_attn_layers,
        self_attn_layers=self_attn_layers,
        pc_padding_val=pc_padding_val, dropout=dropout)
    tot_num_params = sum(p.numel() for p in geo_encoder.parameters())
    trainable_params = sum(p.numel()
                           for p in geo_encoder.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of Geo encoder: {tot_num_params}, {trainable_params} of which are trainable")
    return geo_encoder


def ImplicitSDFModelDefinition(latent_d=128, d_hidden_sdfnn=[128, 128]):
    # latent_d have to be the same as the latent_d of the geo_encoder
    sdf_NN = implicit_sdf(latent_d=latent_d, d_hidden_sdfnn=d_hidden_sdfnn)
    tot_num_params = sum(p.numel() for p in sdf_NN.parameters())
    trainable_params = sum(p.numel()
                           for p in sdf_NN.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of SDF NN: {tot_num_params}, {trainable_params} of which are trainable")
    return sdf_NN

def EvaluateGeoEncoderModel(trainer, test_loader, grid_coor, sdf_inv_scaler):
    sd_pred, sd_true = trainer.predict(test_loader, grid_coor)
    sd_pred = sdf_inv_scaler(sd_pred)
    sd_true = sdf_inv_scaler(sd_true)
    error_s = np.linalg.norm(sd_pred-sd_true, axis=1) / \
        np.linalg.norm(sd_true, axis=1)
    mean, std = np.mean(error_s), np.std(error_s)
    print(f"Mean L2 error of SDF: {mean}, std: {std}")
    sort_idx = np.argsort(error_s)
    min_index = sort_idx[0]
    max_index = sort_idx[int(len(sort_idx)*0.97)]
    median_index = sort_idx[len(sort_idx) // 2]
    print("Index for minimum geo:", min_index,
          "with error", error_s[min_index])
    print("Index for 97 percentile geo:", max_index,
          "with error", error_s[max_index])
    print("Index for median geo:", median_index,
          "with error", error_s[median_index])


def TrainGeoEncoderModel(geo_encoder, sdf_NN, filebase, train_flag, epochs=300, lr=1e-3):

    train_dataset, test_dataset, grid_coor, sdf_inv_scaler, stress_inv_scaler = LoadData(
        seed=42)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    grid_coor = grid_coor.to(device)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            pc = data[0].to(self.device)
            SDF = data[2].to(self.device)
            params = self.models[0](pc)
            sdf_pred = self.models[1](grid_coor, params)
            loss = nn.MSELoss()(sdf_pred, SDF)
            loss_dic = {"loss": loss.item()}
            return loss, loss_dic

        def predict(self, data, grid_coor):
            sd_pred = []
            sd_true = []
            self.models[0].eval()
            self.models[1].eval()
            with torch.no_grad():
                for data in data:
                    pc = data[0].to(self.device)
                    SDF = data[2].to(self.device)
                    params = self.models[0](pc)
                    sdf_pred = self.models[1](grid_coor, params)
                    sd_pred.append(sdf_pred.cpu().detach().numpy())
                    sd_true.append(SDF.cpu().detach().numpy())
            sd_pred = np.vstack(sd_pred)
            sd_true = np.vstack(sd_true)
            return sd_pred, sd_true

    trainer = TRAINER(
        {"encoder": geo_encoder, "sdf_NN": sdf_NN}, device, filebase)

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
                    epochs=epochs)

    trainer.load_weights(device=device)
    trainer.save_logs()
    EvaluateGeoEncoderModel(trainer, test_loader, grid_coor, sdf_inv_scaler)
    return trainer


def LoadGeoEncoderModel(file_base, model_args):
    geo_encoder, sdf_NN = GeoEncoderModelDefinition(**model_args)
    geo_encoder_path = os.path.join(file_base, "encoder", "model.ckpt")
    sdf_NN_path = os.path.join(file_base, "sdf_NN", "model.ckpt")
    for model, path in zip([geo_encoder, sdf_NN], [geo_encoder_path, sdf_NN_path]):
        state_dict = torch.load(
            path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    return geo_encoder, sdf_NN


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start", help="start or continue, or any other string")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))
    configs = models_configs()
    filebase = configs["GeoEncoder"]["filebase"]
    model_args = configs["GeoEncoder"]["model_args"]
    print(f"\n\nGeoEncoder Filebase: {filebase}, model_args:")
    print(model_args)
    geo_encoder, sdf_NN = GeoEncoderModelDefinition(**model_args)
    trainer = TrainGeoEncoderModel(geo_encoder, sdf_NN, filebase, args.train_flag,
                                   epochs=args.epochs, lr=args.learning_rate)
    print(filebase, " training finished")
# %%
# geo_encoder.eval()
# pc = torch.randn(2, 300, 2)
# padv = -10
# pc[0, 180:, :] = padv
# pc[1, 220:, :] = padv
# geo_encoder.pc_padding_val = padv
# params = geo_encoder(pc)
# params = params.detach().cpu().numpy()
# # %%
# # padv = -200

# # pc[0, 180:, :] = padv
# # geo_encoder.pc_padding_val = padv
# pc_new = pc[:, :240, :]
# seed = torch.randint(0, 100000, (1,)).item()
# # seed = 43646
# torch.manual_seed(seed)
# print("Seed for random permutation: ", seed)
# pc_new = pc_new[:, torch.randperm(pc_new.size(1)), :]
# params_pad = geo_encoder(pc_new, apply_padding_pointnet2=True)

# params_pad = params_pad.detach().cpu().numpy()
# for i in range(len(params)):
#     L2 = np.linalg.norm(params[i] - params_pad[i])/np.linalg.norm(params[i])

#     print("L2 norm of params with and without padding", L2)


# %%
