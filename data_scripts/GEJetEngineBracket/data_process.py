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
import pickle
pv.set_jupyter_backend('html')
# pv.set_jupyter_backend('client')
# pv.global_theme.trame.server_proxy_enabled = False
# pv.global_theme.trame.jupyter_extension_enabled=True

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
data_path = f"{SCRIPT_PATH}/../../data/GEJetEngineBracket/"
field_mesh_path = f"{data_path}/FieldMesh/"
volume_mesh_path = f"{data_path}/VolumeMesh/"
# %%
nodal_variable_name = "ver_stress(MPa)"
"""nodal_var:
['1st_mode_shape_resultant', '1st_mode_shape_x', '1st_mode_shape_y',
'1st_mode_shape_z', '2nd_mode_shape_resultant', '2nd_mode_shape_x',
'2nd_mode_shape_y', '2nd_mode_shape_z', 'dia_resultant_disp(mm)',
'dia_stress(MPa)', 'dia_x_disp(mm)', 'dia_y_disp(mm)', 'dia_z_disp(mm)',
'hor_resultant_disp(mm)', 'hor_stress(MPa)', 'hor_x_disp(mm)',
'hor_y_disp(mm)', 'hor_z_disp(mm)', 'tor_resultant_disp(mm)',
'tor_stress(MPa)', 'tor_x_disp(mm)', 'tor_y_disp(mm)', 'tor_z_disp(mm)',
'ver_resultant_disp(mm)', 'ver_stress(MPa)', 'ver_y_disp(mm)',
'ver_z_disp(mm)']>"""
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


def extract_data(index):
    with h5py.File(field_mesh_files[index], 'r') as f:
        nodal_variable = f['nodal_variables'][nodal_variable_name][:].astype(
            np.float32)
    # add zeros for the 5 loading points
    nodal_variable = np.append(nodal_variable, [0]*5)
    # Load the VTK file using pyvista
    vtk_mesh = pv.read(volume_mesh_files[index])
    # original_celltypes = vtk_mesh.celltypes
    original_points = vtk_mesh.points
    # 10-node tetrahedral elements
    original_cells = vtk_mesh.cells.reshape(-1, 11)
    # check the last 5 nodes are keypoints for loading
    node_ids = np.unique(original_cells)
    assert np.all(node_ids == np.arange(len(node_ids)))
    assert len(node_ids) == len(original_points)-5
    # reduce 10 node tetrahedral elements to 4 node tetrahedral elements
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
    # new_celltypes = np.full(len(corrected_cells), pv.CellType.TETRA)
    new_nodal_variable = nodal_variable[used_points]
    new_celltypes = np.full(len(corrected_cells), pv.CellType.TETRA)
    new_mesh = pv.UnstructuredGrid(corrected_cells, new_celltypes, new_points)
    surface_mesh = new_mesh.extract_surface()
    surface_points = surface_mesh.points
    return new_points, corrected_cells, new_nodal_variable, surface_points
# %%


new_points, corrected_cells, new_nodal_variable, surface_points = extract_data(
    2)
# %%


def pv_plot(points, cells, nodal_variable, surface_points):
    # cells:(n_cells, 5)
    celltypes = np.full(len(cells), pv.CellType.TETRA)
    mesh = pv.UnstructuredGrid(cells, celltypes, points)
    mesh.point_data[nodal_variable_name] = nodal_variable
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, cmap="rainbow", show_edges=True)
    plotter.show()
    # Plot the surface mesh
    plotter = pv.Plotter()
    plotter.add_mesh(surface_points, color="lightblue", show_edges=True)
    plotter.show()


# %%
vertices_all = []
cells_all = []
nodal_stress_all = []
points_cloud_all = []
for idx in range(len(sample_ids)):
    print(f"========Processing {idx}: {sample_ids[idx]}=============")
    points, cells, nodal_variable, surface_points = extract_data(
        idx)
    vertices_all.append(new_points)
    cells_all.append(corrected_cells)
    nodal_stress_all.append(new_nodal_variable)
    points_cloud_all.append(surface_points)
# %%
data = {"vertices": vertices_all, "cells": cells_all,
        "nodal_stress": nodal_stress_all, "points_cloud": points_cloud_all}
with open(os.path.join(data_path, 'GE-JEB.pkl'), 'wb') as f:
    pickle.dump(data, f)
