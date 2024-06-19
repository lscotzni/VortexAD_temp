import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh, gen_panel_mesh_new
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
import matplotlib.pyplot as plt

from VortexAD.utils.plot import plot_wireframe, plot_pressure_distribution
from VortexAD import SAMPLE_GEOMETRY_PATH

import pyvista as pv

b = 10.
# c = 1.564
c = 1.
ns = 5
nc = 11

alpha_deg = 0.
alpha = np.deg2rad(alpha_deg) # aoa

mach = 0.3
sos = 340.3
# V_inf = np.array([sos*mach, 0., 0.])
V_inf = np.array([-10., 0., 0.])
nt = 10
num_nodes = 1

mesh_orig = gen_panel_mesh(nc, ns, c, b, span_spacing='default',  frame='default', plot_mesh=False)
# mesh_orig = gen_panel_mesh_new(nc, ns, c, b,  frame='default', plot_mesh=True)
# mesh_orig[:,:,1] += 5.
# exit()

# filename = str(SAMPLE_GEOMETRY_PATH) + '/pm/wing_NACA0012_ar10.vtk' 
# mesh_data = pv.read(filename)
# mesh_orig = mesh_data.points.reshape((21,5,3))

mesh = np.zeros((num_nodes, nt) + mesh_orig.shape)
for i in range(num_nodes):
    for j in range(nt):
        mesh[i,j,:] = mesh_orig

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocities = np.zeros_like(mesh)
for i in range(num_nodes):
    for j in range(nt):
        mesh_velocities[i,j,:] = V_inf_rot

recorder = csdl.Recorder(inline=True)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocities = csdl.Variable(value=mesh_velocities)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocities]

output_dict, mesh_dict, wake_mesh_dict, gamma, gamma_wake = unsteady_panel_solver(mesh_list, mesh_velocity_list, dt=0.05, mode='vortex-ring', free_wake=False)


mesh = mesh_dict['surface_0']['mesh'].value
coll_points = mesh_dict['surface_0']['panel_center'].value
Cp = output_dict['surface_0']['Cp'].value
# CL = output_dict['surface_0']['CL'].value

# CL_norm = csdl.norm(output_dict['surface_0']['CL'])

# dCL_dmesh = csdl.derivative(CL_norm, mesh_velocities)

recorder.stop()

CL  = output_dict['surface_0']['CL'].value
CDi = output_dict['surface_0']['CDi'].value

print('doublet distribution:')
print(gamma.value[0,0,:].reshape((nc-1)*2,ns-1))
print(f'CL: {CL}')
print(f'CDi: {CDi}')

# exit()


verif = True
if verif and alpha_deg == 0.:

    chord_station = coll_points[0,0,:,int((ns-1)/2),0]
    chord = mesh[0,0,0,int((ns-1)/2),0]
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

    plt.plot(LHJ_data_Re9[:,0], LHJ_data_Re9[:,1], '^r', label='Ladson et al. ')
    plt.plot(Gregory_data[:,0], Gregory_data[:,1], '>g', label='Gregory et al. ')
    plt.plot(chord_station/chord, Cp_station, 'k*', label='CSDL panel code')

    plt.gca().invert_yaxis()
    plt.xlabel('Normalized chord')
    plt.ylabel('$C_p$')
    plt.legend()
    plt.grid()
    plt.show()

if verif and alpha_deg == 10.:
    chord_station = coll_points[0,0,:,int((ns-1)/2),0]
    chord = mesh[0,0,0,int((ns-1)/2),0]
    Cp_station = Cp[0,-2,:,int((ns-1)/2)]

    LHJ_data_Re9 = np.array([
    [.9483,   .1147],
    [.9000,   .0684],
    [.8503,   .0882],
    [.7998,   .0849],
    [.7497,   .0782],
    [.7003,   .0739],
    [.6502,   .0685],
    [.5997,   .0813],
    [.5506,   .0884],
    [.5000,   .0940],
    [.4503,   .1125],
    [.4000,   .1225],
    [.3507,   .1488],
    [.3002,   .1893],
    [.2501,   .2292],
    [.2004,   .2973],
    [.1504,   .3900],
    [.1000,   .5435],
    [.0755,   .6563],
    [.0510,   .8031],
    [.0251,  1.0081],
    [.0122,  1.0241],
    [0.   , -2.6598],
    [.0135, -3.9314],
    [.0271, -3.1386],
    [.0515, -2.4889],
    [.0763, -2.0671],
    [.1012, -1.8066],
    [.1503, -1.4381],
    [.1994, -1.2297],
    [.2501, -1.0638],
    [.2999,  -.9300],
    [.3499,  -.8094],
    [.3994,  -.7131],
    [.4496,  -.6182],
    [.4997,  -.5374],
    [.5492,  -.4563],
    [.5994,  -.3921],
    [.6495,  -.3247],
    [.6996,  -.2636],
    [.7489,  -.1964],
    [.8003,  -.1318],
    [.8500,  -.0613],
    [.8993,  -.0021],
    [.9489,   .0795],
    ])

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    plt.plot(LHJ_data_Re9[:,0], LHJ_data_Re9[:,1], '^r', label='Ladson et al. ')
    plt.plot(chord_station/chord, Cp_station, 'k*', label='CSDL panel code')

    plt.gca().invert_yaxis()
    plt.xlabel('Normalized chord')
    plt.ylabel('$C_p$')
    plt.legend()
    plt.grid()
    plt.show()
1
wake_mesh = wake_mesh_dict['surface_0']['mesh'].value

if False:
    plot_pressure_distribution(mesh, Cp, interactive=True, top_view=False)

if False:
    # plot_wireframe(mesh, wake_mesh, gamma.value, gamma_wake.value, nt, interactive=False, backend='cv', name=f'wing_vr_{alpha_deg}')
    plot_wireframe(mesh, wake_mesh, gamma.value, gamma_wake.value, nt, interactive=False, backend='cv', name='vr_free_wake_demo')
