import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
import matplotlib.pyplot as plt
import time
# from VortexAD.utils.plot import plot_wireframe

from VortexAD.utils.cell_adjacency import find_cell_adjacency

from VortexAD import SAMPLE_GEOMETRY_PATH
import meshio

b = 10
c = 1.564
ns = 11
nc = 5

alpha = np.deg2rad(0.) # aoa

mach = 0.25
sos = 340.3
V_inf = np.array([sos*mach, 0., 0.])
nt = 5
num_nodes = 1

# points_orig, connectivity = gen_panel_mesh(nc, ns, c, b, frame='default', unstructured=True, plot_mesh=False)
# exit()

file_name = str(SAMPLE_GEOMETRY_PATH) + '/pm/naca0012_mesh.msh'
mesh = meshio.read(
    file_name,  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)
# NOTE: THE THREE LISTS BELOW USE 1-BASE INDEXING; NEED TO SHIFT DOWN BY 1
upper_TE_cells = np.array([260, 263, 265, 266, 268, 270, 272, 275, 276, 261]) - 112 - 1
lower_TE_cells = np.array([119, 123, 125, 127, 129, 131, 132, 135, 137, 122]) - 112 - 1
TE_node_indices = np.array([1, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9]) - 1

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict

triangles = cells_dict['triangle']
lines = cells_dict['line']
points_orig, triangles, cell_adjacency = find_cell_adjacency(points=points_orig, cells=triangles)

points = np.zeros((num_nodes, nt) + points_orig.shape)
for i in range(num_nodes):
    for j in range(nt):
        points[i,j,:] = points_orig

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

point_velocities = np.zeros_like(points)
for i in range(num_nodes):
    for j in range(nt):
        point_velocities[i,j,:] = V_inf_rot

recorder = csdl.Recorder(inline=True)
recorder.start()

points = csdl.Variable(value=points)
point_velocities = csdl.Variable(value=point_velocities)
TE_data = [TE_node_indices, (upper_TE_cells, lower_TE_cells)]

connectivity_data = [triangles, cell_adjacency]

start_time = time.time()
output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(
    points, 
    connectivity_data, 
    TE_data, 
    point_velocities, 
    dt=0.01, 
    mesh_mode='unstructured'
)
run_stop = time.time()

CL = output_dict['surface_0']['CL']

# dCL_dmesh = csdl.derivative(csdl.norm(CL),mesh)
deriv_stop = time.time()

print('run time: ', run_stop - start_time)
print('deriv time: ', deriv_stop - run_stop)
# print(dCL_dmesh)
recorder.stop()

recorder.print_graph_structure()
recorder.visualize_graph(filename='test_graph', visualize_style='hierarchical')

exit()

mesh = mesh_dict['surface_0']['mesh'].value
coll_points = mesh_dict['surface_0']['panel_center'].value
Cp = output_dict['surface_0']['Cp'].value

chord_station = coll_points[0,0,:,int((ns-1)/2),0]
chord = chord_station[0]
Cp_station = Cp[0,0,:,int((ns-1)/2)]

LHJ_data_Re6 = np.array([
[.9483,   .1008],
[.9000,   .0279],
[.8503,  -.0038],
[.7998,  -.0378],
[.7497,  -.0731],
[.7003,  -.1027],
[.6502,  -.1428],
[.5997,  -.1585],
[.5506,  -.1887],
[.5000,  -.2152],
[.4503,  -.2371],
[.4000,  -.2716],
[.3507,  -.2958],
[.3002,  -.3257],
[.2501,  -.3550],
[.2004,  -.3854],
[.1504,  -.3986],
[.1000,  -.3949],
[.0755,  -.3815],
[.0510,  -.3522],
[.0251,  -.2208],
[.0122,   .0070],
[0.,     1.0184],
[.0135,   .0407],
[.0271,  -.1745],
[.0515,  -.2864],
[.0763,  -.3605],
[.1012,  -.3644],
[.1503,  -.3592],
[.1994,  -.3618],
[.2501,  -.3346],
[.2999,  -.3139],
[.3499,  -.2805],
[.3994,  -.2606],
[.4496,  -.2300],
[.4997,  -.2065],
[.5492,  -.1737],
[.5994,  -.1494],
[.6495,  -.1278],
[.6996,  -.0967],
[.7489,  -.0698],
[.8003,  -.0371],
[.8500,   .0011],
[.8993,   .0477],
[.9489,   .0973],
])

LHJ_data_Re9 = np.array([
[.9483,   .0957],
[.9000,   .0196],
[.8503,  -.0124],
[.7998,  -.0450],
[.7497,  -.0790],
[.7003,  -.1147],
[.6502,  -.1540],
[.5997,  -.1781],
[.5506,  -.1885],
[.5000,  -.2314],
[.4503,  -.2544],
[.4000,  -.2863],
[.3507,  -.3151],
[.3002,  -.3401],
[.2501,  -.3761],
[.2004,  -.4026],
[.1504,  -.4267],
[.1000,  -.4250],
[.0755,  -.4032],
[.0510,  -.3743],
[.0251,  -.2033],
[.0122,   .0147],
[0.   ,  1.0416],
[.0135,   .0138],
[.0271,  -.2018],
[.0515,  -.3335],
[.0763,  -.3923],
[.1012,  -.4084],
[.1503,  -.3926],
[.1994,  -.3904],
[.2501,  -.3702],
[.2999,  -.3446],
[.3499,  -.3085],
[.3994,  -.2814],
[.4496,  -.2527],
[.4997,  -.2241],
[.5492,  -.1783],
[.5994,  -.1743],
[.6495,  -.1430],
[.6996,  -.1117],
[.7489,  -.0765],
[.8003,  -.0462],
[.8500,  -.0092],
[.8993,   .0364],
[.9489,   .0901],
])

Gregory_data = np.array([
[0, 1],
[0.0023497, 0.847673],
[0.00496048, 0.456198],
[0.00526903, 0.173569],
[0.0142406, -0.044407],
[0.0209337, -0.175278],
[0.0473501, -0.372653],
[0.0779437, -0.396388],
[0.0976194, -0.41941],
[0.128166, -0.418874],
[0.150001, -0.411087],
[0.178387, -0.402938],
[0.289702, -0.36672],
[0.322431, -0.347115],
[0.387891, -0.307906],
[0.448983, -0.268412],
[0.514442, -0.229203],
[0.579902, -0.189994],
[0.638834, -0.159098],
[0.704317, -0.114629],
[0.767593, -0.065278],
[0.835236, -0.026211],
[0.896305, 0.03502],
[0.959533, 0.0978565],
[1.0009, 0.173854],
])

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

plt.plot(chord_station/chord, Cp_station, 'k-*', label='CSDL panel code')
plt.plot(LHJ_data_Re9[:,0], LHJ_data_Re9[:,1], '^g', label='Ladson et al. ')
plt.plot(Gregory_data[:,0], Gregory_data[:,1], '>r', label='Gregory et al. ')
plt.gca().invert_yaxis()
plt.xlabel('Normalized chord')
plt.ylabel('$C_p$')
plt.legend()
plt.grid()
plt.show()

1
# if False:
#     plot_wireframe(mesh, wake_mesh, mu_b, mu_wake, nt, interactive=False)
