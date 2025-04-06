import csdl_alpha as csdl
import numpy as np 
from VortexAD import steady_panel_solver

# plotting functions
import matplotlib.pyplot as plt
from VortexAD.utils.plot_unstructured import plot_pressure_distribution
from VortexAD import SAMPLE_GEOMETRY_PATH

from VortexAD.utils.check_duplicate_nodes import check_duplicate_nodes
from VortexAD.utils.cell_adjacency import find_cell_adjacency
from VortexAD.utils.TE_detection import TE_detection
from VortexAD.utils.get_TE_data import get_TE_data
import meshio

alpha_deg = 0.
alpha = np.deg2rad(alpha_deg) # aoa
mach = 0.4
sos = 340.3
# V_inf = np.array([-sos*mach, 0., 0.])
V_inf = np.array([-100., 0., 0.])
num_nodes = 1

# file_name = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_LE_TE_cluster.stl'
# file_name = str(SAMPLE_GEOMETRY_PATH) + '/pm/ngal_coarse.stl'
file_name = str(SAMPLE_GEOMETRY_PATH) + '/pm/ngal_coarse_TE_90.stl'
# file_name = str(SAMPLE_GEOMETRY_PATH) + '/pm/ngal.stl'
mesh = meshio.read(
    file_name,  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict

triangles = cells_dict['triangle']

# dup_indices = check_duplicate_nodes(points=points_orig)
# print(dup_indices)

# exit()
points_orig, triangles, cell_adjacency, edges2cells, points2cells = find_cell_adjacency(points=points_orig, cells=triangles)
# edges2cells = find_cell_adjacency(points=points_orig, cells=triangles)

upper_TE_cells, lower_TE_cells, TE_edges, TE_node_indices = TE_detection(
    points=points_orig,
    cells=triangles,
    # cell_adjacency=cell_adjacency,
    edges2cells=edges2cells
)

# upper_TE_cells, lower_TE_cells, TE_node_indices = get_TE_data(points_orig, triangles, cell_adjacency, edges2cells)

TE_coloring = np.zeros(shape=triangles.shape[0])
TE_coloring[upper_TE_cells] = 1
TE_coloring[lower_TE_cells] = -1

# plot_pressure_distribution(points_orig, TE_coloring, connectivity=triangles, interactive=True, top_view=False, cmap='rainbow')
# exit()
element_colors = np.zeros(shape=(triangles.shape[0],))
# element_colors[5] = 1
# element_colors = np.arange(triangles.shape[0])

# bad_elements = []
# for i, key in enumerate(edges2cells.keys()):
#     if len(edges2cells[key]) < 2:
#         cell_ind = edges2cells[key][0]
#         element_colors[cell_ind] = 1
# plot_pressure_distribution(points_orig, element_colors, connectivity=triangles, interactive=True, top_view=False)

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
# exit()
recorder = csdl.Recorder(inline=False, debug=True)
recorder.start()
dummy_input = csdl.Variable(value=1.)
points = csdl.Variable(value=points) * dummy_input
point_velocities = csdl.Variable(value=point_velocities)
# TE_data = [TE_node_indices, TE_edges, (upper_TE_cells, lower_TE_cells)]
TE_data = [TE_node_indices, TE_edges, (lower_TE_cells, upper_TE_cells)]

connectivity_data = [triangles, cell_adjacency, points2cells]

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
# AIC_mu_orig = output_dict['AIC_mu_orig']
AIC_mu = output_dict['AIC_mu']
use_jax = True
if use_jax:
    recorder.print_largest_variables()
    # exit()
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[points],
        # additional_inputs=[dummy_input],
        additional_outputs = [mu, Cp, CL, CDi]
        # additional_outputs = [CL, CDi]
        # additional_outputs = [Cp]
    )
    # exit()
    jax_sim.run()
    # jax_sim.check_totals(step_size=1.e-3)
    # exit()
    CL = jax_sim[CL]
    CDi = jax_sim[CDi]
    points = jax_sim[points]
    Cp = jax_sim[Cp]
    mu = jax_sim[mu]
    # AIC_mu_orig = jax_sim[AIC_mu_orig]
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

if True:
    plot_pressure_distribution(points[0,:], Cp[0,:], bounds=[-2, 1], connectivity=triangles, interactive=True, top_view=False)
    # plot_pressure_distribution(points[0,:], Cp[0,:], connectivity=triangles, interactive=True, top_view=False)
exit()
num_panels = triangles.shape[0]
AIC_ind = np.arange(num_panels)
# AIC_self_int = AIC_mu_orig[0,AIC_ind, AIC_ind]

# AIC_int_p0 = AIC_mu_orig[0,0,:]

if False:
    plot_pressure_distribution(points[0,:], mu[0,:], connectivity=triangles, interactive=True, top_view=False)
    # plot_pressure_distribution(points[0,:], AIC_self_int, connectivity=triangles, interactive=True, top_view=False)
    # plot_pressure_distribution(points[0,:], AIC_int_p0, connectivity=triangles, interactive=True, top_view=False)


# debuggin pressure dist
Cp_abs = np.abs(Cp[0,:])

error_ind = list(np.where(Cp_abs > 10)[0])

Cp_good_bad = np.zeros(shape=Cp_abs.shape)
Cp_good_bad[list(error_ind)] = 1
if True:
    plot_pressure_distribution(points[0,:], Cp_good_bad, connectivity=triangles, interactive=True, top_view=False, cmap='cool')

'''
NOTE-s:
- can plot pressure values in a histogram with "n" bins
    - good way to quantify how much of the simulation is in the right ballpark
'''