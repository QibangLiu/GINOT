# %%
"""
Instructions:
-- Download the required data and place them in ../../data/GEJetEngineBracket/:
    VolumeMesh, FieldMesh and BRep can be downloaded from the following link:
    https://drive.google.com/drive/u/0/folders/10ccsas7TfD7nIan-Ll5y9vbx4-tSqEyJ
-- Important Notes:
** Do not use the FieldMesh for mesh.
    The point, cell, and face data in the FieldMesh should not be used.
    As of February 8, 2025, there is a mismatch between the corresponding nodes in cells and fields due to the removal of the 5 loading points.
    They may correct this issue in the future, but for now, the FieldMesh should not be used.
** Recommended Approach:
    Run data_process_brep.py first, to extract The point cloud of surface from the BRep files (*.step) in the BRep folder, using freecad
    which has smaller size than the surface mesh extracted from the VTK files.

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
import gmsh
pv.set_jupyter_backend('html')
# pv.set_jupyter_backend('client')
# pv.global_theme.trame.server_proxy_enabled = False
# pv.global_theme.trame.jupyter_extension_enabled=True

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
data_path = f"{SCRIPT_PATH}/../../data/GEJetEngineBracket/"
field_mesh_path = f"{data_path}/FieldMesh/"
volume_mesh_path = f"{data_path}/VolumeMesh/"
brep_path = f"{data_path}/BRep/"
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
brep_files = [os.path.join(brep_path, id+".step") for id in sample_ids]
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


def gmsh_extract_surface_points(index):
    # has issues for some *.step files
    file = brep_files[index]
    gmsh.initialize()
    gmsh.open(file)
    model = gmsh.model
    model_geo = model.geo

    model_geo.synchronize()
    entities = model.getEntities(0)

    points = []
    for entity in entities:
        points.append(model.getValue(0, entity[1], []))

    points = np.array(points)
    gmsh.finalize()
    return points




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
    # new_celltypes = np.full(len(corrected_cells), pv.CellType.TETRA)
    # new_mesh = pv.UnstructuredGrid(corrected_cells, new_celltypes, new_points)
    # surface_mesh = new_mesh.extract_surface()
    # surface_points = surface_mesh.points
    # surface_points = extract_surface_points(index)
    return new_points, corrected_cells, new_nodal_variable

# new_points, corrected_cells, new_nodal_variable = extract_data(
#     2)
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
points_cloud_dic = np.load(f"{data_path}points_cloud_dict.npz")
for idx in range(len(sample_ids)):
    print(f"========Processing {idx}: {sample_ids[idx]}=============")
    points, cells, nodal_variable = extract_data(
        idx)
    points = points.view(np.ndarray)
    surface_points = points_cloud_dic[sample_ids[idx]]
    vertices_all.append(points.astype(np.float32))
    cells_all.append(cells)
    nodal_stress_all.append(nodal_variable.astype(np.float32))
    points_cloud_all.append(surface_points.astype(np.float32))
# %%
data = {"vertices": vertices_all, "cells": cells_all,
        "nodal_stress": nodal_stress_all, "points_cloud": points_cloud_all}
with open(os.path.join(data_path, 'GE-JEB.pkl'), 'wb') as f:
    pickle.dump(data, f)
