import csdl_alpha as csdl
import numpy as np 
from VortexAD import steady_panel_solver
import jax
jax.config.update("jax_enable_x64", True)

# plotting functions
import matplotlib.pyplot as plt
from VortexAD.utils.plot_unstructured import plot_pressure_distribution
from VortexAD import SAMPLE_GEOMETRY_PATH

from VortexAD.utils.check_duplicate_nodes import check_duplicate_nodes
from VortexAD.utils.cell_adjacency import find_cell_adjacency
from VortexAD.utils.TE_detection import TE_detection
from VortexAD.utils.get_TE_data import get_TE_data
import meshio
import time

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
points_orig = points

# exit()
recorder = csdl.Recorder(inline=False, debug=True)
recorder.start()

x_scaler = csdl.Variable(value=np.array([1.]))
y_scaler = csdl.Variable(value=np.array([1.]))
z_scaler = csdl.Variable(value=np.array([1.]))

points = csdl.Variable(value=points_orig)
points = points.set(csdl.slice[:,:,0], points_orig[:,:,0]*x_scaler)
points = points.set(csdl.slice[:,:,1], points_orig[:,:,1]*y_scaler)
points = points.set(csdl.slice[:,:,2], points_orig[:,:,2]*z_scaler)

V_inf = csdl.Variable(value=np.array([-100.]))
pitch = csdl.Variable(value=np.array([0.0]))
pitch_rad = pitch*np.pi/180

V_vec = csdl.Variable(value=0., shape=(3,))
V_vec = V_vec.set(csdl.slice[0], value=V_inf)

V_rot_mat = csdl.Variable(value=0., shape=(3,3))
V_rot_mat = V_rot_mat.set(csdl.slice[1,1], value=1.)
V_rot_mat = V_rot_mat.set(csdl.slice[0,0], value=csdl.cos(pitch_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[2,2], value=csdl.cos(pitch_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[2,0], value=csdl.sin(pitch_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[0,2], value=-csdl.sin(pitch_rad))

V_vec_rot = csdl.matvec(V_rot_mat, V_vec)

point_velocities = csdl.expand(V_vec_rot, points.shape, 'i->abi')

# TE_data = [TE_node_indices, TE_edges, (upper_TE_cells, lower_TE_cells)]
TE_data = [TE_node_indices, TE_edges, (lower_TE_cells, upper_TE_cells)]
connectivity_data = [triangles, cell_adjacency, points2cells]

output_dict, mesh_dict, mu, sigma = steady_panel_solver(
    points, 
    connectivity_data, 
    TE_data, 
    point_velocities, 
    mesh_mode='unstructured',
    batch_size=1
)

CL = output_dict['CL']
CDi = output_dict['CDi']
Di = output_dict['Di']
L = output_dict['L']
coll_points = mesh_dict['panel_center']
Cp = output_dict['Cp']
# AIC_mu_orig = output_dict['AIC_mu_orig']
AIC_mu = output_dict['AIC_mu']

moment = output_dict['M']

pitch_moment = moment[:,1]

dM_dpitch = csdl.derivative(pitch_moment, pitch)

recorder.print_largest_variables()
# exit()

inputs = [
    x_scaler,
    y_scaler,
    z_scaler,
    pitch,
    V_inf
]

inputs = [pitch]

check_derivatives = True
if check_derivatives:
    outputs = [L]
    outputs = [moment, dM_dpitch]
else:
    outputs = [points, mu, Cp, L, Di]

jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs,
    additional_outputs=outputs
)

# jaxified_panel_func = csdl.jax.create_jax_function(
#     graph = recorder.get_root_graph(),
#     outputs = outputs,
#     inputs = [dummy_input]
# )
# import jax.numpy as jnp
# import jax
# jax.config.update("jax_enable_x64", True)
# jax_out = jax.jit(jaxified_panel_func)(jnp.array([1.0]))

# jax_derivatives = jax.jit(jax.jacrev(jaxified_panel_func))(jnp.array([1.0]))

# exit()
print('dummy compile + run of forward evaluation')
jax_sim.run()
print('end dummy run')

print('running forward eval')
start_time = time.time()
jax_sim.run()
end_time = time.time()
print(f'forward eval run time: {end_time-start_time} seconds')

if check_derivatives:
    print('dummy compile + run of derivatives')
    jax_sim.compute_totals()
    print('end dummy run')

    print('running derivatives')
    start_time = time.time()
    jax_sim.check_totals(step_size=1.e-3)
    # jax_sim.compute_totals()
    end_time = time.time()
    print(f'derivative run time: {end_time-start_time} seconds')
    exit()

L = jax_sim[L]
Di = jax_sim[Di]
points = jax_sim[points]
Cp = jax_sim[Cp]
mu = jax_sim[mu]
# AIC_mu_orig = jax_sim[AIC_mu_orig]


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