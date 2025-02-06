# %%
import trimesh
import time
import os
import pickle
import pyvista
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import LinearProblem
from ufl import TrialFunction, TestFunction, dx, inner, grad, lhs, rhs
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc, form)
from basix.ufl import element
from mpi4py import MPI
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
import gmsh
import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

# %%
theta = np.linspace(0, 1, 145)
model = gs.Gaussian(dim=1, var=10, len_scale=0.15)


def run_sim():
    srf = gs.SRF(
        model,
        generator="Fourier",
        period=1.0,
        mode_no=32,
        seed=None
    )
    r = srf(theta, mesh_type="structured")
    r = (r-np.min(r))/(np.max(r)-np.min(r))*0.8+0.2
    xb = np.cos(theta*np.pi*2)*r
    yb = np.sin(theta*np.pi*2)*r
    xy_bound = np.vstack([xb[:-1], yb[:-1]]).T
    # plt.plot(xy_bound[:, 0], xy_bound[:, 1])
    # gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)  # Lower verbosity level

    gmsh.model.add("loop")
    lc = 0.05          # mesh element size (adjust as needed)
    point_tags = []

    for xy in xy_bound:
        tag = gmsh.model.geo.addPoint(xy[0], xy[1], 0, lc)
        point_tags.append(tag)
    line_tags = []
    numPoints = len(point_tags)
    for i in range(numPoints):
        start = point_tags[i]
        end = point_tags[(i + 1) % numPoints]  # wrap around to the first point
        line = gmsh.model.geo.addLine(start, end)
        line_tags.append(line)
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    gmsh.model.geo.synchronize()
    phy_tag = 1
    boundary = gmsh.model.addPhysicalGroup(
        1, line_tags, phy_tag, name="boundary")
    phy_tag += 1
    surface_group = gmsh.model.addPhysicalGroup(
        2, [surface], phy_tag, name='Interior')
    phy_tag += 1
    # Synchronize the built geometry with the gmsh model

    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()  # Uncomment to view mesh interactively

    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    msh, cell_tags, ft = gmshio.model_to_mesh(
        gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"
    cell_tags.name = "Cell markers"
    fdim = msh.topology.dim-1
    sca_element = element("Lagrange", msh.topology.cell_name(), 1)
    V = functionspace(msh, sca_element)
    boundary_facets = inlet_facets = ft.find(1)
    boundary_dofs = locate_dofs_topological(
        V, fdim, ft.find(1))
    bc = dirichletbc(0.0, boundary_dofs, V)
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)
    f = 1.0
    a, L = 0.1*inner(grad(u), grad(v))*dx, f*v*dx
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={
                            "ksp_type": "preonly", "pc_type": "lu"})
    u_solution = problem.solve()
    gmsh.finalize()
    pyvista_cells, cell_types, nodes = vtk_mesh(V)
    return pyvista_cells, cell_types, nodes, u_solution.x.array, xy_bound
# %%
# visualize


def plot_pyvista(pyvista_cells, cell_types, nodes, u_solution, jupyter_backend='client'):
    pyvista.global_theme.trame.server_proxy_enabled = True
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, nodes)
    grid.point_data["u"] = u_solution
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=False)
    plotter.view_xy()
    try:
        plotter.show(jupyter_backend=jupyter_backend)
    except:
        pass


def plot_trimesh(nodes, pyvista_cells, u_solution):
    cells = pyvista_cells.reshape(-1, 4)[:, 1:]
    # mesh = trimesh.Trimesh(vertices=nodes, faces=cells)
    # mesh.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = ax.tricontourf(nodes[:, 0], nodes[:, 1], cells,
                       u_solution, levels=20, cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_title('Contour of u_solution')


def test():
    pyvista_cells, cell_types, nodes, u_solution, xy_bound = run_sim()
    plot_pyvista(pyvista_cells, cell_types, nodes, u_solution)


# %%
def run_all():
    cells_all = []
    celltypes_all = []
    nodes_all = []
    solutions_all = []
    point_clouds_all = []
    start_time = time.time()
    while len(cells_all) < 6001:
        try:
            pyvista_cells, cell_types, nodes, u_solution, xy_bound = run_sim()
        except:
            print("failed")
            continue
        cells_all.append(pyvista_cells)
        celltypes_all.append(cell_types)
        nodes_all.append(nodes.astype(np.float32))
        solutions_all.append(u_solution.astype(np.float32))
        point_clouds_all.append(xy_bound.astype(np.float32))
        print(
            f"got {len(cells_all)} samples, took {time.time()-start_time} seconds")

    data_all = {'cells': cells_all, 'celltypes': celltypes_all, 'nodes': nodes_all,
                'solutions': solutions_all, 'point_clouds': point_clouds_all}
    filebase = "/work/nvme/bbka/qibang/repository_WNbbka/GINTO_data/poisson/"

    os.makedirs(filebase, exist_ok=True)
    with open(filebase+'poisson_geo.pkl', 'wb') as f:
        pickle.dump(data_all, f)
    # return data_all
