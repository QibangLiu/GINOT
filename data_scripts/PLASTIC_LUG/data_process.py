# %%
import pyvista as pv
from collections import Counter
import natsort
import h5py
import pickle
import numpy as np
import os
import pickle
pv.set_jupyter_backend('html')
# %%
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
data_path = f"{SCRIPT_PATH}/../../data/PLASTIC_LUG/"
origi_data_path = "/projects/bblv/skoric/PLASTIC_LUG_PARAM_VAR_THICK"
vtk_path = f"{origi_data_path}/Input_Mesh/vtks"
sample_ids = np.arange(0, 3000)
# %%
vtk_files = os.listdir(vtk_path)
vtk_files = natsort.natsorted(vtk_files)
vtk_files = [f"{vtk_path}/{f}" for f in vtk_files]

# %%
stress_file = f"{origi_data_path}/Extraction_from_Odb/stress_targets.npz"
stress_all = np.load(stress_file)
# %%


def extract_data(index):
    nodal_variable = 1e-6 * stress_all['Job_' + str(index)]
    # Load the VTK file using pyvista
    vtk_mesh = pv.read(vtk_files[index])
    vtk_mesh.point_data["nodal_stress"] = nodal_variable
    # original_celltypes = vtk_mesh.celltypes
    original_points = vtk_mesh.points
    # 20-node QUADRATIC_HEXAHEDRON
    original_cells = vtk_mesh.cells.reshape(-1, 21)
    # reduce 10 node tetrahedral elements to 4 node tetrahedral elements
    reduced_cells = []
    for cell in original_cells:
        new_cell = cell[1:9]
        new_cell = [8] + list(new_cell)
        reduced_cells.append(new_cell)
    reduced_cells = np.array(reduced_cells)
    used_points = np.unique(reduced_cells[:, 1:])
    used_points = np.sort(used_points)
    point_mapping = {old_idx: new_idx for new_idx,
                     old_idx in enumerate(used_points)}
    new_points = original_points[used_points, :]
    corrected_cells = []
    for cell in reduced_cells:
        new_cell = [8] + [point_mapping[old_idx] for old_idx in cell[1:]]
        corrected_cells.append(new_cell)
    corrected_cells = np.array(corrected_cells)
    new_celltypes = np.full(len(corrected_cells), pv.CellType.HEXAHEDRON)
    new_nodal_variable = nodal_variable[used_points]
    new_mesh = pv.UnstructuredGrid(
        corrected_cells.flatten(), new_celltypes, new_points)
    surface_mesh = new_mesh.extract_surface()
    surface_points = surface_mesh.points
    return new_points, corrected_cells, new_celltypes, new_nodal_variable, surface_points


# %%
new_points, corrected_cells, new_celltypes, new_nodal_variable, surface_points = extract_data(
    0)


def pv_plot(new_points, corrected_cells, new_celltypes, new_nodal_variable, surface_points):
    new_mesh = pv.UnstructuredGrid(
        corrected_cells.flatten(), new_celltypes, new_points)
    plotter = pv.Plotter()
    new_mesh["nodal_stress"] = new_nodal_variable
    plotter.add_mesh(new_mesh, cmap="jet", show_edges=False)
    # plotter.add_mesh(vtk_mesh, cmap="jet", show_edges=False)
    plotter.show()


# %%
vertices_all = []
cells_all = []
nodal_stress_all = []
points_cloud_all = []
for idx in range(len(vtk_files)):
    print(f"========Processing {idx}: {sample_ids[idx]}=============")
    points, cells, celltypes, stress, surface_points = extract_data(
        idx)
    points = points.view(np.ndarray).astype(np.float32)
    surface_points = surface_points.view(np.ndarray).astype(np.float32)
    stress = stress.squeeze().astype(np.float32)
    vertices_all.append(points)
    cells_all.append(cells)
    nodal_stress_all.append(stress)
    points_cloud_all.append(surface_points)
# %%
data = {"vertices": vertices_all,
        "nodal_stress": nodal_stress_all, "points_cloud": points_cloud_all}
with open(os.path.join(data_path, 'LUG_node_S_PC.pkl'), 'wb') as f:
    pickle.dump(data, f)

with open(os.path.join(data_path, 'LUG_cells.pkl'), 'wb') as f:
    pickle.dump(cells_all, f)
# %%
