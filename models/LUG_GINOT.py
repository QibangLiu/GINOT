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
    from .modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from .modules.UNets import UNet
    from .trainer import torch_trainer
    from .modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from .modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
else:
    import configs
    from modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from modules.UNets import UNet
    from trainer import torch_trainer
    from modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                                         nn.Linear(2*embed_dim, 3*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(3*embed_dim, 3*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(3*embed_dim, 2*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(2*embed_dim, out_channels)
                                         )

        self.loading_encoder = nn.Sequential(nn.Linear(1, 2*embed_dim),
                                             nn.SiLU(),
                                             nn.Linear(
                                                 2*embed_dim, 2*embed_dim),
                                             nn.SiLU(),
                                             nn.Linear(2*embed_dim, embed_dim),
                                             nn.SiLU(),
                                             )
        self.latent_encoder = nn.Sequential(nn.Linear(2*embed_dim, 2*embed_dim),
                                            nn.SiLU(),
                                            nn.Linear(2*embed_dim, embed_dim),
                                            nn.SiLU()
                                            )

    def forward(self, load, xyt, pc, sample_ids=None):
        # (B, 1)->(B, embed_dim)
        load = self.loading_encoder(load)
        # (B, np, embed_dim)
        latent = self.branch(pc, sample_ids=sample_ids)
        B, np, _ = latent.shape
        load = load.view(B, 1, -1).expand(-1, np, -1)
        latent = torch.cat([latent, load], dim=-1)

        latent = self.latent_encoder(latent)
        # (B,N,ndim)->(B,N,embed_dim)
        xyt = encode_position('nerf', position=xyt)
        x = self.Q_encoder(xyt)

        for block in self.resblocks:
            x = block(x, latent)  # (B, N, embed_dim)
        # (B, N, embed_dim)->(B, N, 1)
        x = self.output_proj(x)
        return x.squeeze(-1)


# %%

def NOTModelDefinition(branch_args, trunc_args):
    branch = PointCloudPerceiverChannelsEncoder(**branch_args)
    tot_num_params = sum(p.numel() for p in branch.parameters())
    trainable_params = sum(p.numel()
                           for p in branch.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of Geo encoder: {tot_num_params}, {trainable_params} of which are trainable")
    trunk = Trunk(branch, **trunc_args)
    tot_num_params = sum(p.numel() for p in trunk.parameters())
    trainable_params = sum(p.numel()
                           for p in trunk.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of NOT model: {tot_num_params}, {trainable_params} of which are trainable")

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


def TrainNOTModel(NTO_model, filebase, train_flag, epochs=300, lr=1e-3, window_size=None):

    train_loader, test_loader, _, s_inverse, pc_inverse, vert_inverse = configs.LoadDataLUGGeo(
        bs_train=32, bs_test=32)

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            load = data[0].to(self.device)
            pc = data[1].to(self.device)
            xyt = data[2].to(self.device)
            y_true = data[3].to(self.device)
            sample_ids = data[4].to(self.device)
            mask = (y_true != self.models[0].padding_value).float()
            y_pred = self.models[0](load, xyt, pc)
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
                    load = data[0].to(self.device)
                    pc = data[1].to(self.device)
                    xyt = data[2].to(self.device)
                    y_true_batch = data[3].to(self.device)
                    mask = (y_true_batch != self.models[0].padding_value)
                    pred = self.models[0](load, xyt, pc)
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
        optimizer, factor=0.7, patience=20)
    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="val_loss",
        window_size=window_size,
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


def LoadModel(filebase, branch_args, trunk_args):
    NTO_model = NOTModelDefinition(branch_args, trunk_args)
    model_path = os.path.join(filebase, "model.ckpt")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    NTO_model.load_state_dict(state_dict)
    NTO_model.to(device)
    NTO_model.eval()
    return NTO_model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=None)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    configs_ginot = configs.LUG_GINOT_configs()

    filebase = configs_ginot["filebase"]
    trunk_args = configs_ginot["trunk_args"]
    branch_args = configs_ginot["branch_args"]
    print(configs_ginot)

    NTO_model = NOTModelDefinition(branch_args, trunk_args)

    trainer = TrainNOTModel(NTO_model, filebase, args.train_flag,
                            epochs=args.epochs, lr=args.learning_rate,
                            window_size=args.window_size)
    print(filebase, " training finished")


# %%
