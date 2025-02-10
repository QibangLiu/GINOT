
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
from sklearn.model_selection import train_test_split


# %%

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_FILEBASE = f"{SCRIPT_PATH}/../data"
DATA_FILE = f"{DATA_FILEBASE}/node_pc_mises_disp_laststep_aug.pkl"


PADDING_VALUE = -10
NUM_POINT_POINTNET2 = 128


def LoadData(data_file=DATA_FILE, test_size=0.2, seed=42):
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    SU_raw = data["mises_disp"]
    su_shift = data['shift']  # (1,3)
    su_scaler = data['scaler']  # (1,3)
    coords = data['mesh_coords']  # (Nb, N, 3)
    points_cloud = data['points_cloud']  # (Nb, N, 3)
    x_grids = data['x_grids'].astype(np.float32)
    y_grids = data['y_grids'].astype(np.float32)

    SU = [torch.tensor((su-su_shift)/su_scaler) for su in SU_raw]  # (Nb, N, 3)
    su_shift = torch.tensor(su_shift)[None, :]  # (1,1,3)
    su_scaler = torch.tensor(su_scaler)[None, :]  # (1, 1, 3)
    xyt = [torch.tensor(x[:, :2]) for x in coords]  # (Nb, N, 2)
    points_cloud = [torch.tensor(x[:, :2])
                    for x in points_cloud]  # (Nb,N,3)->(Nb, N, 2)
    num_p = [x.shape[0] for x in points_cloud]
    if min(num_p) < NUM_POINT_POINTNET2:
        raise ValueError(
            f"Number of sample points {NUM_POINT_POINTNET2}\
            should be smaller than the minimum number of points in the point cloud {min(num_p)}")

    SU = pad_sequence(SU, batch_first=True, padding_value=PADDING_VALUE)
    xyt = pad_sequence(xyt, batch_first=True, padding_value=PADDING_VALUE)
    points_cloud = pad_sequence(
        points_cloud, batch_first=True, padding_value=PADDING_VALUE)
    grid_coor = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
    grid_coor = torch.tensor(grid_coor)

    train_pc, test_pc, train_xyt, test_xyt, train_su, test_su = train_test_split(
        points_cloud, xyt, SU, test_size=test_size, random_state=seed)

    train_dataset = TensorDataset(
        train_pc, train_xyt, train_su)
    test_dataset = TensorDataset(test_pc, test_xyt, test_su)

    def SUInverse(x):
        # x: (Nb, N, 3), mises stress, ux, uy
        if isinstance(x, torch.Tensor):
            shift = su_shift.to(x.device)
            scaler = su_scaler.to(x.device)
        else:
            shift = su_shift.cpu().numpy()
            scaler = su_scaler.cpu().numpy()
        return x*scaler+shift
    su_inverse = SUInverse
    return train_dataset, test_dataset, grid_coor, su_inverse


# %%


def models_configs(out_c=256, latent_d=256, *args, **kwargs):
    """************GeoEncoder arguments************"""

    fps_method = "fps"
    geo_encoder_model_args = {
        "out_c": out_c,
        "latent_d": latent_d,
        "width": 128,
        "n_point": NUM_POINT_POINTNET2,
        "n_sample": 8,
        "radius": 0.2,
        "d_hidden": [128, 128],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 3,
        "pc_padding_val": PADDING_VALUE,
        "d_hidden_sdfnn": [128, 128],
        "fps_method": fps_method,
    }
    geo_encoder_file_base = f"{SCRIPT_PATH}/saved_weights/geoencoder_outc{out_c}_latentdim{latent_d}_fps{fps_method}"
    geo_encoder_args = {
        "model_args": geo_encoder_model_args,
        "filebase": geo_encoder_file_base,
    }
    """************NTO arguments************"""
    embed_dim = 64
    channel_mutipliers = [1, 2, 4]
    has_attention = [False, False, False]
    first_conv_channels = 16
    num_res_blocks = 1
    norm_groups = None
    dropout = None
    NTO_img_shape = (1, 120, 120)

    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/NTO_outc{out_c}_latentdim{latent_d}_noatt_normgroups-{norm_groups}_dropout-{dropout}_posenc"

    NTO_model_args = {"embed_dim": embed_dim,
                      "img_shape": NTO_img_shape,
                      "channel_mutipliers": channel_mutipliers,
                      "has_attention": has_attention,
                      "first_conv_channels": first_conv_channels,
                      "num_res_blocks": num_res_blocks,
                      "norm_groups": norm_groups,
                      "dropout": dropout,
                      "padding_value": PADDING_VALUE, }
    NTO_args = {"model_args": NTO_model_args, "filebase": NTO_filebase}

    args_all = {"GeoEncoder": geo_encoder_args,
                "NTOModel": NTO_args, }
    return args_all


# %%
def LoadDataPoissionGeo(test_size=0.2, seed=42):
    data_file = f"{DATA_FILEBASE}/poisson/poisson_geo.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    cells_all = data['cells']
    cells_all = [c.reshape(-1, 4)[:, 1:] for c in cells_all]
    nodes_all = data['nodes']
    solutions_all = data['solutions']
    point_clouds_all = data['point_clouds']
    points_cloud = torch.tensor(np.array(point_clouds_all))
    solutions = [torch.tensor(s) for s in solutions_all]
    nodes = [torch.tensor(n[:, :2]) for n in nodes_all]
    solutions = pad_sequence(solutions, batch_first=True,
                             padding_value=PADDING_VALUE)
    xyt = pad_sequence(nodes, batch_first=True, padding_value=PADDING_VALUE)
    train_pc, test_pc, train_xyt, test_xyt, train_u, test_u, train_idx, test_idx = train_test_split(
        points_cloud, xyt, solutions, np.arange(0, len(xyt)), test_size=test_size, random_state=seed)

    train_dataset = TensorDataset(
        train_pc, train_xyt, train_u)
    test_dataset = TensorDataset(test_pc, test_xyt, test_u)
    test_cells = [cells_all[i] for i in test_idx]
    return train_dataset, test_dataset, test_cells


def poission_geo_from_pc_configs():

    fps_method = "fps"
    out_c = 64
    geo_encoder_model_args = {
        "out_c": out_c,
        "latent_d": 64,
        "width": 128,
        "n_point": 36,
        "n_sample": 18,
        "radius": 0.2,
        "d_hidden": [128, 128],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 3,
        "d_hidden_sdfnn": [128, 128],
        "fps_method": fps_method,
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 4, "padding_value": PADDING_VALUE}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/poission_geo_frompc_test"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all


# %%
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


def elasticity_geo_from_pc_configs():

    fps_method = "fps"
    out_c = 64
    geo_encoder_model_args = {
        "out_c": out_c,
        "latent_d": 64,
        "width": 64,
        "n_point": 36,
        "n_sample": 18,
        "radius": 0.2,
        "d_hidden": [64, 64],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 2,
        "d_hidden_sdfnn": [128, 128],
        "fps_method": fps_method,
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 6}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/elasticity_geo_from_pc_for"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all

# %%
# ==================================ShapeCar====================================


def LoadDataShapeCarGeo():
    data_file = f"{DATA_FILEBASE}/shapeNet-car/pc_pressure_data.npz"
    data = np.load(data_file)
    points_cloud = data['points_cloud'].astype(np.float32)
    nodes = points_cloud.copy()
    # shuffle the points in the point cloud
    for pc in points_cloud:
        np.random.shuffle(pc)
    pressures = data['pressures'].astype(np.float32)

    points_cloud = torch.tensor(points_cloud)
    nodes = torch.tensor(nodes)
    p_shift, p_scale = np.mean(pressures), np.std(pressures)
    p_norm = (pressures-p_shift)/p_scale
    pressures = torch.tensor(p_norm)

    # num_train = 500
    # num_test = 111
    # train_pc = points_cloud[:num_train]
    # train_xyt = nodes[:num_train]
    # train_u = pressures[:num_train]
    # test_pc = points_cloud[-num_test:]
    # test_xyt = nodes[-num_test:]
    # test_u = pressures[-num_test:]
    train_pc, test_pc, train_xyt, test_xyt, train_u, test_u = train_test_split(
        points_cloud, nodes, pressures, test_size=0.18, random_state=42)

    train_dataset = TensorDataset(
        train_pc, train_xyt, train_u)
    test_dataset = TensorDataset(test_pc, test_xyt, test_u)

    def PressureInverse(x):
        return x*p_scale+p_shift
    p_inverse = PressureInverse

    return train_dataset, test_dataset, p_inverse


def shape_car_geo_from_pc_configs():

    fps_method = "first"
    out_c = 32
    dropout = 0.1
    geo_encoder_model_args = {
        "input_channels": 3,
        "out_c": out_c,
        "latent_d": 32,
        "width": 32,
        "n_point": 512,
        "n_sample": 256,
        "radius": 0.4,
        "d_hidden": [64, 64],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 2,
        "d_hidden_sdfnn": [128, 128],
        "fps_method": fps_method,
        "dropout": dropout
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 3, "dropout": dropout}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/shape-car_geo_from_pc_test"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all


# =============================================================================
# %%
# ========================GE Jet Engine Bracket================================
def LoadDataGEJEBGeo(test_size=0.2, seed=42):
    data_file = f"{DATA_FILEBASE}/GEJetEngineBracket/GE-JEB.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    vertices = data['vertices']
    cells = data['cells']
    nodal_stress = data['nodal_stress']
    points_cloud = data['points_cloud']
    vert_concat = np.concatenate(vertices, axis=0, dtype=np.float32)
    vert_shift, vert_scale = np.mean(
        vert_concat, axis=0), np.std(vert_concat, axis=0)
    vert_shift = vert_shift[None, :]
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
    vert_shift = torch.tensor(vert_shift)[None, :]
    vert_scale = torch.tensor(vert_scale)[None, :]

    S = pad_sequence(sigma_norm, batch_first=True,
                     padding_value=PADDING_VALUE)
    points_cloud = pad_sequence(
        pc_norm, batch_first=True, padding_value=PADDING_VALUE)
    vertices = pad_sequence(vertices_norm, batch_first=True,
                            padding_value=PADDING_VALUE)
    train_pc, test_pc, train_xyt, test_xyt, train_u, test_u, train_ids, test_ids = train_test_split(
        points_cloud, vertices, S, np.arange(len(cells)), test_size=test_size, random_state=seed)
    train_dataset = TensorDataset(
        train_pc, train_xyt, train_u)
    test_dataset = TensorDataset(test_pc, test_xyt, test_u)
    cells_test = [cells[i] for i in test_ids]
    cells_train = [cells[i] for i in train_ids]

    def SigmaInverse(x):
        return x*sigma_scale+sigma_shift
    s_inverse = SigmaInverse

    def PCInverse(x):
        pc_scale = pc_scale.to(x.device)
        pc_shift = pc_shift.to(x.device)
        return x*pc_scale+pc_shift
    pc_inverse = PCInverse

    def VertInverse(x):
        vert_scale = vert_scale.to(x.device)
        vert_shift = vert_shift.to(x.device)
        return x*vert_scale+vert_shift
    vert_inverse = VertInverse

    return train_dataset, test_dataset, cells_train, cells_test, s_inverse, pc_inverse, vert_inverse


def JEB_geo_from_pc_configs():
    fps_method = "first"
    out_c = 16
    dropout = 0
    geo_encoder_model_args = {
        "input_channels": 3,
        "out_c": out_c,
        "latent_d": 32,
        "width": 16,
        "n_point": 1024,
        "n_sample": 256,
        "radius": 0.5,
        "d_hidden": [16, 16],
        "num_heads": 4,
        "cross_attn_layers": 1,
        "self_attn_layers": 1,
        "d_hidden_sdfnn": [128, 128],
        "fps_method": fps_method,
        "pc_padding_val": PADDING_VALUE,
        "dropout": dropout
    }
    trunc_model_args = {"embed_dim": out_c,
                        "cross_attn_layers": 1, "num_heads": 4, "dropout": dropout, "padding_value": PADDING_VALUE}
    NTO_filebase = f"{SCRIPT_PATH}/saved_weights/JEB_geo_from_pc_test"
    args_all = {"branch_args": geo_encoder_model_args,
                "trunk_args": trunc_model_args, "filebase": NTO_filebase}
    return args_all

# =============================================================================
# %%
