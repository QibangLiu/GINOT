
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
import pickle
from sklearn.model_selection import train_test_split
from typing import Union
import time
# %%
# Total memory allocated by tensors (in bytes)
# print(f"Allocated memory: {torch.cuda.memory_allocated('cuda') / 1024**3:.2f} GB")
#     # Total memory reserved by PyTorch's caching allocator (in bytes)
# print(f"Reserved memory: {torch.cuda.memory_reserved('cuda') / 1024**3:.2f} GB")
# %%

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_FILEBASE = f"{SCRIPT_PATH}/../data"


PADDING_VALUE = -1000

# %%


class ListDataset(Dataset):
    """for list of tensors"""

    def __init__(self, data: Union[list, tuple]):
        """
        args:
            data: list of data, each element is a list of tensors
            e.g. [(pc1, xyt1, S1), (pc2, xyt2, S2), ...]

        """
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        one_data = [d[idx] for d in self.data]
        return one_data


# %%
# ==================================Poisson equation====================================
def LoadDataPoissonGeo(struct=True, test_size=0.2, seed=42):
    if struct:
        data_file = f"{DATA_FILEBASE}/poisson/poisson_geo_struc_msh.pkl"
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        cells_all = data['cells']
        nodes_all = data['nodes']
        solutions_all = data['solutions']
        point_clouds_all = data['point_clouds']
        xyt = torch.tensor(np.array(nodes_all))
        xyt = xyt[:, :, :2]
        points_cloud = torch.tensor(np.array(point_clouds_all))
        solutions = torch.tensor(np.array(solutions_all))
    else:
        data_file = f"{DATA_FILEBASE}/poisson/poisson_geo_unstruc_msh.pkl"
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        cells_all = data['cells']
        nodes_all = data['nodes']
        solutions_all = data['solutions']
        point_clouds_all = data['point_clouds']
        points_cloud = torch.tensor(np.array(point_clouds_all))
        solutions = [torch.tensor(s) for s in solutions_all]
        nodes = [torch.tensor(n[:, :2]) for n in nodes_all]
        solutions = pad_sequence(solutions, batch_first=True,
                                 padding_value=PADDING_VALUE)
        xyt = pad_sequence(nodes, batch_first=True,
                           padding_value=PADDING_VALUE)

    train_pc, test_pc, train_xyt, test_xyt, train_u, test_u, train_idx, test_idx = train_test_split(
        points_cloud, xyt, solutions, torch.arange(0, len(xyt)), test_size=test_size, random_state=seed)

    train_dataset = TensorDataset(
        train_pc, train_xyt, train_u, train_idx)
    test_dataset = TensorDataset(test_pc, test_xyt, test_u, test_idx)

    return train_dataset, test_dataset, cells_all


def poisson_GINOT_configs(struct=True):

    if struct:
        NTO_filebase = f"{SCRIPT_PATH}/saved_weights/poisson_ginot_struct_msh"
    else:
        NTO_filebase = f"{SCRIPT_PATH}/saved_weights/poisson_ginot_unstruct_msh"

    fps_method = "fps"
    out_c = 64
    geo_encoder_model_args = {
        "out_c": out_c,
        "latent_d": None,
        "width": 128,
        "n_point": 64,
        "n_sample": 18,
        "radius": 0.2,
        "d_hidden": [128, 128],
        "num_heads": 8,
        "cross_attn_layers": 1,
        "self_attn_layers": 3,
        "fps_method": fps_method,
    }
    trunc_model_args = {"embed_dim": out_c, "num_heads": 8,
                        "cross_attn_layers": 4, "padding_value": PADDING_VALUE}
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all


# %%
# ==================================elasticity====================================
def LoadDataElasticityGeo(test_size=0.2, seed=42):
    data_file = f"{DATA_FILEBASE}/elasticity/elasticity.npz"
    data = np.load(data_file)
    points_cloud_all = data['points_cloud'].astype(np.float32)
    nodes_all = data['nodes'].astype(np.float32)
    sigma_all = data['sigma'].astype(np.float32)

    points_cloud = torch.tensor(points_cloud_all)
    nodes = torch.tensor(nodes_all)
    sigma_shift, sigma_scale = np.mean(sigma_all), np.std(sigma_all)
    sigma_norm = (sigma_all-sigma_shift)/sigma_scale
    sigma = torch.tensor(sigma_norm)

    # train_pc, test_pc, train_xyt, test_xyt, train_u, test_u = train_test_split(
    #     points_cloud, nodes, sigma, test_size=test_size, random_state=seed)
    num_train = 1000
    num_test = 200
    train_pc = points_cloud[:num_train]
    train_xyt = nodes[:num_train]
    train_u = sigma[:num_train]
    test_pc = points_cloud[-num_test:]
    test_xyt = nodes[-num_test:]
    test_u = sigma[-num_test:]

    train_dataset = TensorDataset(
        train_pc, train_xyt, train_u)
    test_dataset = TensorDataset(test_pc, test_xyt, test_u)

    def SigmaInverse(x):
        return x*sigma_scale+sigma_shift
    s_inverse = SigmaInverse

    return train_dataset, test_dataset, s_inverse


def elasticity_GINOT_configs():

    fps_method = "fps"
    out_c = 64
    geo_encoder_model_args = {
        "out_c": out_c,
        "latent_d": None,
        "width": 64,
        "n_point": 64,
        "n_sample": 18,
        "radius": 0.2,
        "d_hidden": [64, 64],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 2,
        "fps_method": fps_method,
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 6}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/elasticity_GINOT"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all

# =============================================================================
# %%
# ========================microstruc Unit Cell ================================


def LoadDataMicroSturcGeo(bs_train=32, bs_test=128, test_size=0.2, seed=42, num_frames=1, load_cells="NO", padding_value=PADDING_VALUE):
    start = time.time()
    if load_cells != "NO":
        mesh_file = f"{DATA_FILEBASE}/PeriodUnitCell/mesh_cells10K.pkl"
        with open(mesh_file, "rb") as f:
            cells = pickle.load(f)
        if load_cells == "ONLY":
            return cells
    elif load_cells == "NO":
        cells = "Nothing"

    if num_frames == 1:
        """load the last frame data"""
        su_file = f"{DATA_FILEBASE}/PeriodUnitCell/mises_disp_laststep.pkl"
        with open(su_file, "rb") as f:
            su_data = pickle.load(f)
            SU_raw = su_data['mises_disp']
            su_shift = su_data['shift']
            su_scaler = su_data['scaler']
    elif num_frames == 11 or num_frames == 26:
        """load the 11 frames data or 26 frames data"""
        scaler_file = f"{DATA_FILEBASE}/PeriodUnitCell/SU_scalers_{num_frames}fram.npz"
        su_scalers = np.load(scaler_file)
        su_shift, su_scaler = su_scalers["global_shift"], su_scalers["global_scale"]
        SU_file = f"{DATA_FILEBASE}/PeriodUnitCell/mises_disp{num_frames}fram.pkl"
        with open(SU_file, "rb") as f:
            SU_raw = pickle.load(f)

    else:
        raise ValueError("num_frames must be 1, 11 or 26")

    pc_file = f"{DATA_FILEBASE}/PeriodUnitCell/points_cloud.pkl"
    with open(pc_file, "rb") as f:
        point_cloud = pickle.load(f)

    node_file = f"{DATA_FILEBASE}/PeriodUnitCell/mesh_coords.pkl"
    with open(node_file, "rb") as f:
        coords = pickle.load(f)

    SU = [torch.tensor((su-su_shift)/su_scaler)
          for su in SU_raw]  # (Nb, N, 3) or (Nb,N,Nt,3 )
    su_shift = torch.tensor(su_shift)[None, :]  # (1,1,3) or (1,1,1,3)
    su_scaler = torch.tensor(su_scaler)[None, :]  # (1, 1, 3) or (1, 1, 1, 3)
    xyt = [torch.tensor(x[:, :2]) for x in coords]  # (Nb, N, 2)
    point_cloud = [torch.tensor(x[:, :2])
                   for x in point_cloud]  # (Nb,N,3)->(Nb, N, 2)
    # split test and train data
    train_ids, test_ids = train_test_split(
        np.arange(len(point_cloud)), test_size=test_size, random_state=seed)
    # test_ids = np.arange(0, 10000)
    # train_ids = np.arange(10000, len(point_cloud))
    train_pc = [point_cloud[i] for i in train_ids]
    test_pc = [point_cloud[i] for i in test_ids]
    train_xyt = [xyt[i] for i in train_ids]
    test_xyt = [xyt[i] for i in test_ids]
    train_S = [SU[i] for i in train_ids]
    test_S = [SU[i] for i in test_ids]
    train_dataset = ListDataset(
        (train_pc, train_xyt, train_S, torch.tensor(train_ids)))
    test_dataset = ListDataset(
        (test_pc, test_xyt, test_S, torch.tensor(test_ids)))

    def pad_collate_fn(batch):
        pc_batch = [item[0] for item in batch]  # Extract pc (variable-length)
        xyt_batch = [item[1]
                     for item in batch]  # Extract xyt (variable-length)
        S = [item[2] for item in batch]  # Extract S (variable-length)
        sample_ids = torch.stack([item[3] for item in batch])
        # y_batch = torch.stack([item[1] for item in batch])  # Extract and stack y (fixed-length)
        # Pad sequences
        pc_padded = pad_sequence(
            pc_batch, batch_first=True, padding_value=padding_value)
        xyt_padded = pad_sequence(
            xyt_batch, batch_first=True, padding_value=padding_value)
        S_padded = pad_sequence(S, batch_first=True,
                                padding_value=padding_value)
        return pc_padded, xyt_padded, S_padded, sample_ids
    train_dataloader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True,
                                  collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False,
                                 collate_fn=pad_collate_fn)

    def SUInverse(x):
        su_sig = su_scaler.to(x.device)
        su_mu = su_shift.to(x.device)
        return x*su_sig+su_mu
    su_inverse = SUInverse
    print(f"Data loading time: {time.time()-start:.2f} s")
    return train_dataloader, test_dataloader, cells, su_inverse


def microstruc_GINOT_configs():
    fps_method = "fps"
    out_c = 128
    dropout = 0.0
    geo_encoder_model_args = {
        "input_channels": 2,
        "out_c": out_c,
        "latent_d": None,
        "width": 128,
        "n_point": 128,
        "n_sample": 8,
        "radius": 0.2,
        "d_hidden": [128, 128],
        "num_heads": 4,
        "cross_attn_layers": 2,
        "self_attn_layers": 2,
        "fps_method": fps_method,
        "pc_padding_val": PADDING_VALUE,
        "dropout": dropout,
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 4, "num_heads": 8, "dropout": dropout, "padding_value": PADDING_VALUE}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/microstruc_laststep_GINOT"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all


def microstruc_multiFrames_GINOT_configs(num_frames=26):
    fps_method = "fps"
    out_c = 128
    dropout = 0.0
    geo_encoder_model_args = {
        "input_channels": 2,
        "out_c": out_c,
        "latent_d": None,
        "width": 128,
        "n_point": 128,
        "n_sample": 8,
        "radius": 0.2,
        "d_hidden": [128, 128],
        "num_heads": 4,
        "cross_attn_layers": 2,
        "self_attn_layers": 2,
        "fps_method": fps_method,
        "pc_padding_val": PADDING_VALUE,
        "dropout": dropout,
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 4, "num_heads": 8, "num_frames": num_frames, "dropout": dropout, "padding_value": PADDING_VALUE}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/microstruc_{num_frames}frames_GINOT"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all
# =============================================================================
# %%
# ========================GE Jet Engine Bracket================================


def LoadDataJEBGeo(bs_train=32, bs_test=128, test_size=0.1, seed=42, padding_value=PADDING_VALUE, shuffle_loader=True):
    start = time.time()
    # load data
    data_file = f"{DATA_FILEBASE}/GEJetEngineBracket/GE-JEB.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    vertices = data['vertices']
    cells = data['cells']
    nodal_stress = data['nodal_stress']
    points_cloud = data['points_cloud']
    # normalize data
    vert_concat = np.concatenate(vertices, axis=0, dtype=np.float32)
    vert_shift, vert_scale = np.mean(
        vert_concat, axis=0), np.std(vert_concat, axis=0)
    vert_shift = vert_shift[None, :]  # (1,3)
    vert_scale = vert_scale[None, :]

    pc_concat = np.concatenate(points_cloud, axis=0, dtype=np.float32)
    pc_shift, pc_scale = np.mean(pc_concat, axis=0), np.std(pc_concat, axis=0)
    pc_shift = pc_shift[None, :]
    pc_scale = pc_scale[None, :]

    sigma_concat = np.concatenate(nodal_stress, axis=0, dtype=np.float32)
    sigma_shift, sigma_scale = np.mean(
        sigma_concat), np.std(sigma_concat)

    vertices_norm = [torch.tensor(
        (v.astype(np.float32)-vert_shift)/vert_scale) for v in vertices]
    pc_norm = [torch.tensor((pc.astype(np.float32)-pc_shift)/pc_scale)
               for pc in points_cloud]
    sigma_norm = [torch.tensor((s.astype(np.float32)-sigma_shift)/sigma_scale)
                  for s in nodal_stress]
    pc_shift = torch.tensor(pc_shift)[None, :]
    pc_scale = torch.tensor(pc_scale)[None, :]
    vert_shift = torch.tensor(vert_shift)[None, :]  # (1, 1,3)
    vert_scale = torch.tensor(vert_scale)[None, :]
    # split test and train data
    train_ids, test_ids = train_test_split(
        np.arange(len(cells)), test_size=test_size, random_state=seed)
    train_pc = [pc_norm[i] for i in train_ids]
    test_pc = [pc_norm[i] for i in test_ids]
    train_xyt = [vertices_norm[i] for i in train_ids]
    test_xyt = [vertices_norm[i] for i in test_ids]
    train_S = [sigma_norm[i] for i in train_ids]
    test_S = [sigma_norm[i] for i in test_ids]
    # cells_test = [cells[i] for i in test_ids]
    # cells_train = [cells[i] for i in train_ids]
    train_dataset = ListDataset(
        (train_pc, train_xyt, train_S, torch.tensor(train_ids)))
    test_dataset = ListDataset(
        (test_pc, test_xyt, test_S, torch.tensor(test_ids)))
    # padded dataloader

    def pad_collate_fn(batch):
        pc_batch = [item[0] for item in batch]  # Extract pc (variable-length)
        xyt_batch = [item[1]
                     for item in batch]  # Extract xyt (variable-length)
        S = [item[2] for item in batch]  # Extract S (variable-length)
        sample_ids = torch.stack([item[3] for item in batch])
        # y_batch = torch.stack([item[1] for item in batch])  # Extract and stack y (fixed-length)
        # Pad sequences
        pc_padded = pad_sequence(
            pc_batch, batch_first=True, padding_value=padding_value)
        xyt_padded = pad_sequence(
            xyt_batch, batch_first=True, padding_value=padding_value)
        S_padded = pad_sequence(S, batch_first=True,
                                padding_value=padding_value)
        return pc_padded, xyt_padded, S_padded, sample_ids

    train_dataloader = DataLoader(train_dataset, batch_size=bs_train, shuffle=shuffle_loader,
                                  collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False,
                                 collate_fn=pad_collate_fn)

    def SigmaInverse(x):
        return x*sigma_scale+sigma_shift
    s_inverse = SigmaInverse

    def PCInverse(x):
        pc_scale_ = pc_scale.to(x.device)
        pc_shift_ = pc_shift.to(x.device)
        return x*pc_scale_+pc_shift_
    pc_inverse = PCInverse

    def VertInverse(x):
        vert_scale_ = vert_scale.to(x.device)
        vert_shift_ = vert_shift.to(x.device)
        return x*vert_scale_+vert_shift_
    vert_inverse = VertInverse
    print(f"Data loading time: {time.time()-start:.2f} s")
    return train_dataloader, test_dataloader, cells, s_inverse, pc_inverse, vert_inverse


def JEB_GINOT_configs():
    fps_method = "fps"
    out_c = 128
    dropout = 0.0
    geo_encoder_model_args = {
        "input_channels": 3,
        "out_c": out_c,
        "latent_d": None,
        "width": 128,
        "n_point": 512,
        "n_sample": 64,
        "radius": 0.5,
        "d_hidden": [128, 128],
        "num_heads": 1,
        "cross_attn_layers": 1,
        "self_attn_layers": 2,
        "fps_method": fps_method,
        "pc_padding_val": PADDING_VALUE,
        "dropout": dropout,
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 3, "num_heads": 1, "dropout": dropout, "padding_value": PADDING_VALUE}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/JEB_GINOT"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all

# =============================================================================
# %%
# ======================Plastic LUG============================


def LoadDataLUGGeo(bs_train=32, bs_test=128, test_size=0.2, seed=42, padding_value=PADDING_VALUE, shuffle_loader=True):
    start = time.time()
    # load loading data
    input_par_file = f"{DATA_FILEBASE}/PLASTIC_LUG/input_params.npy"
    inp_para = np.load(input_par_file)
    loading = inp_para[:, 3].astype(np.float32)
    loading = (loading-loading.mean())/loading.std()
    loading = torch.tensor(loading).view(-1, 1)
    # load cells
    cells_file = f"{DATA_FILEBASE}/PLASTIC_LUG/LUG_cells.pkl"
    with open(cells_file, "rb") as f:
        cells = pickle.load(f)
    # load data
    data_file = f"{DATA_FILEBASE}/PLASTIC_LUG/LUG_node_S_PC.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    vertices = data['vertices']
    nodal_stress = data['nodal_stress']
    points_cloud = data['points_cloud']
    # normalize data
    vert_concat = np.concatenate(vertices, axis=0, dtype=np.float32)
    vert_shift, vert_scale = np.mean(
        vert_concat, axis=0), np.std(vert_concat, axis=0)
    vert_shift = vert_shift[None, :]  # (1,3)
    vert_scale = vert_scale[None, :]

    pc_concat = np.concatenate(points_cloud, axis=0, dtype=np.float32)
    pc_shift, pc_scale = np.mean(pc_concat, axis=0), np.std(pc_concat, axis=0)
    pc_shift = pc_shift[None, :]
    pc_scale = pc_scale[None, :]

    sigma_concat = np.concatenate(nodal_stress, axis=0, dtype=np.float32)
    sigma_shift, sigma_scale = np.mean(
        sigma_concat), np.std(sigma_concat)

    vertices_norm = [torch.tensor(
        (v.astype(np.float32)-vert_shift)/vert_scale) for v in vertices]
    pc_norm = [torch.tensor((pc.astype(np.float32)-pc_shift)/pc_scale)
               for pc in points_cloud]
    sigma_norm = [torch.tensor((s.astype(np.float32)-sigma_shift)/sigma_scale)
                  for s in nodal_stress]
    pc_shift = torch.tensor(pc_shift)[None, :]
    pc_scale = torch.tensor(pc_scale)[None, :]
    vert_shift = torch.tensor(vert_shift)[None, :]  # (1, 1,3)
    vert_scale = torch.tensor(vert_scale)[None, :]
    # split test and train data
    train_ids, test_ids = train_test_split(
        np.arange(len(vertices)), test_size=test_size, random_state=seed)
    train_loading = loading[train_ids]
    test_loading = loading[test_ids]
    train_pc = [pc_norm[i] for i in train_ids]
    test_pc = [pc_norm[i] for i in test_ids]
    train_xyt = [vertices_norm[i] for i in train_ids]
    test_xyt = [vertices_norm[i] for i in test_ids]
    train_S = [sigma_norm[i] for i in train_ids]
    test_S = [sigma_norm[i] for i in test_ids]
    # cells_test = [cells[i] for i in test_ids]
    # cells_train = [cells[i] for i in train_ids]
    train_dataset = ListDataset(
        (train_loading, train_pc, train_xyt, train_S, torch.tensor(train_ids)))
    test_dataset = ListDataset(
        (test_loading, test_pc, test_xyt, test_S, torch.tensor(test_ids)))
    # padded dataloader

    def pad_collate_fn(batch):
        loading = torch.stack([item[0] for item in batch])
        pc_batch = [item[1] for item in batch]  # Extract pc (variable-length)
        xyt_batch = [item[2]
                     for item in batch]  # Extract xyt (variable-length)
        S = [item[3] for item in batch]  # Extract S (variable-length)
        sample_ids = torch.stack([item[4] for item in batch])
        # y_batch = torch.stack([item[1] for item in batch])  # Extract and stack y (fixed-length)
        # Pad sequences
        pc_padded = pad_sequence(
            pc_batch, batch_first=True, padding_value=padding_value)
        xyt_padded = pad_sequence(
            xyt_batch, batch_first=True, padding_value=padding_value)
        S_padded = pad_sequence(S, batch_first=True,
                                padding_value=padding_value)
        return loading, pc_padded, xyt_padded, S_padded, sample_ids

    train_dataloader = DataLoader(train_dataset, batch_size=bs_train, shuffle=shuffle_loader,
                                  collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False,
                                 collate_fn=pad_collate_fn)

    def SigmaInverse(x):
        return x*sigma_scale+sigma_shift
    s_inverse = SigmaInverse

    def PCInverse(x):
        pc_scale_ = pc_scale.to(x.device)
        pc_shift_ = pc_shift.to(x.device)
        return x*pc_scale_+pc_shift_
    pc_inverse = PCInverse

    def VertInverse(x):
        vert_scale_ = vert_scale.to(x.device)
        vert_shift_ = vert_shift.to(x.device)
        return x*vert_scale_+vert_shift_
    vert_inverse = VertInverse
    print(f"Data loading time: {time.time()-start:.2f} s")
    return train_dataloader, test_dataloader, cells, s_inverse, pc_inverse, vert_inverse


# %%


def LUG_GINOT_configs():
    fps_method = "fps"
    out_c = 128
    dropout = 0.0
    geo_encoder_model_args = {
        "input_channels": 3,
        "out_c": out_c,
        "latent_d": None,
        "width": 128,
        "n_point": 512,
        "n_sample": 64,
        "radius": 0.5,
        "d_hidden": [128, 128],
        "num_heads": 8,
        "cross_attn_layers": 1,
        "self_attn_layers": 2,
        "fps_method": fps_method,
        "pc_padding_val": PADDING_VALUE,
        "dropout": dropout,
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 3, "num_heads": 8, "dropout": dropout, "padding_value": PADDING_VALUE}
    # NTO_filebase = f"{SCRIPT_PATH}/saved_weights/JEB_GINOT"
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/LUG_GINOT"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all
