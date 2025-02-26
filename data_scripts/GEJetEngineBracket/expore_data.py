# %%
"""
Instructions:
-- Download the required data and place them in ../../data/GEJetEngineBracket/:
    VolumeMesh and FieldMesh can be downloaded from the following link:
    https://drive.google.com/drive/u/0/folders/10ccsas7TfD7nIan-Ll5y9vbx4-tSqEyJ
-- Important Notes:
** Do not use the SurfaceMesh.
    The point, cell, and face data in the FieldMesh should not be used.
    As of February 8, 2025, there is a mismatch between the corresponding nodes in cells and fields due to the removal of the 5 loading points.
** Recommended Approach:
    Use the VTK files in the VolumeMesh folder and the nodal variables from the FieldMesh.
    These two datasets are matched and should be used together.
** Mesh Information:
    The meshes consist of 10-node tetrahedral elements,
    but this script will converte the mesh and data to 4-node tetrahedral elements to reduce the number of nodes.
    This conversion should be treated as a Derivative Database.
    The Derivative Database is governed by the terms specified in the original license file:
    https://drive.google.com/file/d/1S4_DphSGNcIzGvFMOjVRJvyhcC_5-nJG/view?usp=drive_link

"""
import trimesh
import gdown
import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv
from collections import Counter
import natsort
import h5py
pv.set_jupyter_backend('html')
# pv.set_jupyter_backend('client')
# pv.global_theme.trame.server_proxy_enabled = False
# pv.global_theme.trame.jupyter_extension_enabled=True

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
data_path = f"{SCRIPT_PATH}/../../data/GEJetEngineBracket/"
field_mesh_path = f"{data_path}/FieldMesh/"
volume_mesh_path = f"{data_path}/VolumeMesh/"
# %%
# List all files under surface_mesh_path
field_mesh_files = os.listdir(field_mesh_path)
sample_ids = [os.path.splitext(f)[0] for f in field_mesh_files]
field_mesh_files = [os.path.join(field_mesh_path, id+".h5")
                    for id in sample_ids]
volume_mesh_files = [os.path.join(
    volume_mesh_path, id+".vtk") for id in sample_ids]
# %%
# Check if all files exist
missing_files = []
for volum_file, field_file in zip(volume_mesh_files, field_mesh_files):
  if not os.path.exists(volum_file):
    missing_files.append(volum_file)
  if not os.path.exists(field_file):
    missing_files.append(field_file)

if missing_files:
  print("The following files are missing:")
  for file in missing_files:
    print(file)
else:
  print("All files exist.")

# %%
sample_index = sample_ids.index("634_421")
print(
    f"The index of sample_id '634_421' is: {sample_index}, {sample_ids[sample_index]}")

# %%
check_file = sample_index
# Load the first field mesh file using h5py
with h5py.File(field_mesh_files[check_file], 'r') as f:
    print("Keys: %s" % f.keys())
    # List all groups
    nodal_var = f['nodal_variables']
    print("nodal_var: %s" % nodal_var.keys())
# %%
nodal_variable_name = "ver_stress(MPa)"
with h5py.File(field_mesh_files[check_file], 'r') as f:
    nodal_variable = f['nodal_variables'][nodal_variable_name][:].astype(
        np.float32)
# %%
vtk_file = volume_mesh_files[check_file]
# Load the VTK file using pyvista
vtk_mesh = pv.read(vtk_file)
nodal_variable = np.append(nodal_variable, [0]*5)
vtk_mesh.point_data[nodal_variable_name] = nodal_variable
# Plot the VTK mesh
plotter = pv.Plotter()
plotter.add_mesh(vtk_mesh, show_edges=True, cmap="jet")
plotter.show()

# %%
original_celltypes = vtk_mesh.celltypes
original_points = vtk_mesh.points
original_cells = vtk_mesh.cells.reshape(-1, 11)  # 10-node tetrahedral elements

# check the last 5 nodes are keypoints for loading
node_ids = np.unique(original_cells)
np.all(node_ids == np.arange(len(node_ids)))
len(node_ids) == len(original_points)-5
# %%
reduced_cells = []
for cell in original_cells:
    new_cell = cell[1:5]
    new_cell = [4] + list(new_cell)
    reduced_cells.append(new_cell)
reduced_cells = np.array(reduced_cells)
used_points = np.unique(reduced_cells[:, 1:])
used_points = np.sort(used_points)
point_mapping = {old_idx: new_idx for new_idx,
                 old_idx in enumerate(used_points)}
new_points = original_points[used_points, :]
corrected_cells = []
for cell in reduced_cells:
    new_cell = [4] + [point_mapping[old_idx] for old_idx in cell[1:]]
    corrected_cells.append(new_cell)
corrected_cells = np.array(corrected_cells)
new_nodal_variable = nodal_variable[used_points]
# %%
new_celltypes = np.full(len(corrected_cells), pv.CellType.TETRA)
new_mesh = pv.UnstructuredGrid(corrected_cells, new_celltypes, new_points)
new_mesh.point_data[nodal_variable_name] = new_nodal_variable
plotter = pv.Plotter()
# plotter.add_points(new_points, color="red", point_size=5)
plotter.add_mesh(new_mesh, show_edges=True, opacity=1.0, cmap="jet")
plotter.show()


# %%
surface_mesh = new_mesh.extract_surface()
surface_points = surface_mesh.points
plotter = pv.Plotter()
plotter.add_mesh(surface_mesh, color="lightblue", opacity=0.5, show_edges=True)
plotter.show()
plotter = pv.Plotter()
plotter.add_mesh(surface_points, color="lightblue", show_edges=True)
plotter.show()
# %%
points_cloud_dic = np.load(f"{data_path}points_cloud_dict.npz")
surface_points = points_cloud_dic[sample_ids[sample_index]]
plotter = pv.Plotter()
plotter.add_mesh(surface_points, color="green",
                 point_size=8)
plotter.show()

# %%
