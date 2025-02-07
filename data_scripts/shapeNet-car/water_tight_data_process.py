# %%
import glob
import trimesh
import gdown
import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv
from collections import Counter
import natsort
pv.set_jupyter_backend('client')
pv.global_theme.trame.server_proxy_enabled = True
# pv.global_theme.trame.jupyter_extension_enabled=True


def download_data(data_path):
    file_names = ["car-cfd.tgz", "car-pressure-data.zip"]

    # Extract from the shared URL
    file_id = ["1SWU-b4GrfFkWUrvxgEVe4XuU9iVvhrSg",
               "1Vb740MGw7dMN943bRTNvy_f_qFMB9sjX"]
    for i in range(len(file_names)):
        if os.path.exists(data_path + file_names[i]):
            print(f"{file_names[i]} already exists.")
        else:
            print(f"Downloading {file_names[i]}...")
            url = f"https://drive.google.com/uc?id={file_id[i]}"
            gdown.download(url, data_path,)

    current_path = os.getcwd()
    os.chdir(data_path)
    if not os.path.exists("./training_data"):
        os.system("tar -xvzf car-cfd.tgz")
    if not os.path.exists("./car-pressure-data"):
        os.system("unzip car-pressure-data.zip")
    os.chdir(current_path)


data_path = "/work/nvme/bbka/qibang/repository_WNbbka/GINTO_data/shapeNet-car/"
download_data(data_path)

#  %%
data_base = f"{data_path}car-pressure-data/data"
water_tight_ids = np.loadtxt(
    f"{data_path}car-pressure-data/watertight_meshes.txt", dtype=str)
print("number of samples", len(water_tight_ids))
ply_files = [os.path.join(data_base, "mesh_"+ids+".ply")
             for ids in water_tight_ids]
press_files = [os.path.join(data_base, "press_"+ids+".npy")
               for ids in water_tight_ids]

bound_p_idx = np.loadtxt(
    f"{data_path}car-pressure-data/boundary_idxs.txt", dtype=int)
# List of extracted numbers
# %%

mesh = trimesh.load(ply_files[0])
# Show edges and color
mesh.visual.face_colors = [0, 100, 0, 100]  # RGBA
mesh.visual.vertex_colors = [0, 0, 255, 100]  # RGBA
points = mesh.vertices
print("number of points", len(points))
assert len(points) == len(bound_p_idx)
# mesh.show(edges=True)
# Create a point cloud from the vertices
pv_mesh = pv.wrap(mesh)

plotter = pv.Plotter()
# Remove faces to only show points and edges
plotter = pv.Plotter()
plotter.add_points(points, color='b', point_size=10)
plotter.add_mesh(pv_mesh, show_edges=True,
                 edge_color='black', opacity=0.6)  # , style="wireframe", color='green'
plotter.show()
# Extract vertices


# %%
pc = []
for ply_f in ply_files:
    mesh = trimesh.load(ply_f)
    points = mesh.vertices
    pc.append(points)
pressures = []
for press_f in press_files:
    press = np.load(press_f)
    pressures.append(press)
pc = np.array(pc)
pressures = np.array(pressures)
pressures = pressures[:, bound_p_idx]
# %%
data = {"points_cloud": pc, "pressures": pressures}
np.savez(f"{data_path}pc_pressure_data.npz", **data)

# %%
