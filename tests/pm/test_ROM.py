import csdl_alpha as csdl
import numpy as np 
from VortexAD import steady_panel_solver

# plotting functions
from VortexAD.utils.plot_unstructured import plot_pressure_distribution
from VortexAD import SAMPLE_GEOMETRY_PATH
from VortexAD.utils.cell_adjacency import find_cell_adjacency
from VortexAD.utils.get_TE_data import get_TE_data
import meshio

alpha_deg = 10.
alpha = np.deg2rad(alpha_deg) # aoa
mach = 0.1
sos = 340.3
# V_inf = np.array([-sos*mach, 0., 0.])
V_inf = np.array([-10., 0., 0.])
num_nodes = 1

file_name = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl'
mesh = meshio.read(
    file_name,  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict

triangles = cells_dict['triangle']
# exit()
points_orig, triangles, cell_adjacency, edges2cells = find_cell_adjacency(points=points_orig, cells=triangles)

upper_TE_cells, lower_TE_cells, TE_node_indices = get_TE_data(points_orig, triangles, cell_adjacency, edges2cells)

# exit()
points = np.zeros((num_nodes, ) + points_orig.shape)
for i in range(num_nodes):
    points[i,:] = points_orig

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

point_velocities = np.zeros_like(points)
for i in range(num_nodes):
    point_velocities[i,:] = V_inf_rot

recorder = csdl.Recorder(inline=False)
recorder.start()

points = csdl.Variable(value=points)
point_velocities = csdl.Variable(value=point_velocities)
TE_data = [TE_node_indices, (upper_TE_cells, lower_TE_cells)]

connectivity_data = [triangles, cell_adjacency]

output_dict, mesh_dict, mu, sigma = steady_panel_solver(
    points, 
    connectivity_data, 
    TE_data, 
    point_velocities, 
    mesh_mode='unstructured'
)

CL = output_dict['CL']
CDi = output_dict['CDi']
coll_points = mesh_dict['panel_center']
Cp = output_dict['Cp']
AIC_mu = output_dict['AIC_mu']
AIC_sigma = output_dict['AIC_sigma']

use_jax = True
if use_jax:
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[points],
        additional_outputs = [mu, coll_points, Cp, CL, CDi, AIC_mu, AIC_sigma]
    )
    jax_sim.run()
    CL = jax_sim[CL]
    CDi = jax_sim[CDi]
    points = jax_sim[points]
    coll_points = jax_sim[coll_points]
    Cp = jax_sim[Cp]
    mu = jax_sim[mu]

    AIC_mu = jax_sim[AIC_mu]
    AIC_sigma = jax_sim[AIC_sigma]
else:
    points = points.value
    coll_points = coll_points.value
    CL = CL.value
    CDi = CDi.value
    mu = mu.value
    Cp = Cp.value


print('doublet distribution:')
print(mu)
print(f'CL: {CL}')
print(f'CDi: {CDi}')


if False:
    plot_pressure_distribution(points[0,:], Cp[0,:], connectivity=triangles, interactive=True, top_view=False)