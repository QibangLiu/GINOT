# %%
import gdown
import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv
from collections import Counter

# %%

num_cell_nodes = {
    pv.CellType.EMPTY_CELL: 0,
    pv.CellType.VERTEX: 1,
    pv.CellType.LINE: 2,
    pv.CellType.TRIANGLE: 3,
    pv.CellType.PIXEL: 4,
    pv.CellType.QUAD: 4,
    pv.CellType.TETRA: 4,
    pv.CellType.VOXEL: 8,
    pv.CellType.HEXAHEDRON: 8,
    pv.CellType.WEDGE: 6,
    pv.CellType.PYRAMID: 5,
    pv.CellType.PENTAGONAL_PRISM: 10,
    pv.CellType.HEXAGONAL_PRISM: 12,

    # Quadratic elements
    pv.CellType.QUADRATIC_EDGE: 3,
    pv.CellType.QUADRATIC_TRIANGLE: 6,
    pv.CellType.QUADRATIC_QUAD: 8,
    pv.CellType.QUADRATIC_TETRA: 10,
    pv.CellType.QUADRATIC_HEXAHEDRON: 20,
    pv.CellType.QUADRATIC_WEDGE: 15,
    pv.CellType.QUADRATIC_PYRAMID: 13,
    pv.CellType.BIQUADRATIC_QUAD: 9,
    pv.CellType.TRIQUADRATIC_HEXAHEDRON: 27,
    pv.CellType.TRIQUADRATIC_PYRAMID: 19,
    pv.CellType.QUADRATIC_LINEAR_QUAD: 6,
    pv.CellType.QUADRATIC_LINEAR_WEDGE: 12,
    pv.CellType.BIQUADRATIC_QUADRATIC_WEDGE: 18,
    pv.CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON: 24,
    pv.CellType.BIQUADRATIC_TRIANGLE: 7,
    pv.CellType.CUBIC_LINE: 4,
}


def download_data(data_path):
    file_names = ["car-cfd.tgz"]
    # Extract from the shared URL
    file_id = ["1SWU-b4GrfFkWUrvxgEVe4XuU9iVvhrSg"]
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
    os.chdir(current_path)


data_path = "/work/nvme/bbka/qibang/repository_WNbbka/GINTO_data/shapeNet-car/"
download_data(data_path)

# %%

data_base = "/work/nvme/bbka/qibang/repository_WNbbka/GINTO_data/shapeNet-car/training_data/param1/1e54527efa629c37a047cd0a07d473f1"
velo = np.load(data_base+"/velo.npy")
press = np.load(data_base+"/press.npy")
# %%
mesh = pv.read(data_base + "/quadpress_smpl.vtk")

# mesh.plot()
# Get the points, cells, and field data from the mesh
points = mesh.points
cells = mesh.cells
field_data = mesh.point_data
cell_types = mesh.celltypes
print("Cell Types:")
print(cell_types)
print("Points:")
print(points)

print("Cells:")
print(cells)

print("Field Data:")
print(field_data)
# %%

# %%
# Find the boundary points based on the cells
num_nodes = np.array([num_cell_nodes[cell_type]
                     for cell_type in cell_types])

connectivity = []
index = 0
for n in num_nodes:
    connectivity.append(cells[index: index + n])
    index += n
unique_elements = np.unique(cells)
boundary_points = points[unique_elements]

# Plot the boundary points using pyvista
boundary_mesh = pv.PolyData(boundary_points)
boundary_mesh.plot()
# %%
# Get unique elements in a 1D array
unique_elements = np.unique(cells)
print("Unique elements in cells:")
print(unique_elements)

# %%
