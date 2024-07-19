import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh, gen_panel_mesh_new
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
import matplotlib.pyplot as plt

from VortexAD.utils.plot import plot_wireframe, plot_pressure_distribution
from VortexAD import SAMPLE_GEOMETRY_PATH

# import pyvista as pv

b = 10.
# c = 1.564
c = 1.
ns = 16 #
nc = 41 #

alpha_deg = 10.
alpha = np.deg2rad(alpha_deg) # aoa

mach = 0.3
sos = 340.3
# V_inf = np.array([sos*mach, 0., 0.])
# V_inf = np.array([-10., 0., 0.])
V_inf = np.array([-10., 0., 0.])
nt = 20 #
num_nodes = 1

mesh_orig = gen_panel_mesh(nc, ns, c, b, span_spacing='cosine',  frame='default', plot_mesh=False)
# mesh_orig = gen_panel_mesh_new(nc, ns, c, b,  frame='default', plot_mesh=False)
# mesh_orig[:,:,1] += 5.
# exit()

# filename = str(SAMPLE_GEOMETRY_PATH) + '/pm/wing_NACA0012_ar10.vtk'
# nc, ns = 11, 5
# mesh_data = pv.read(filename)
# mesh_orig = mesh_data.points.reshape((2*nc-1,ns,3))
# mesh_orig[:,:,1] -= 5.


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
recorder = csdl.Recorder(inline=False)
recorder.start()
 
mesh = csdl.Variable(value=mesh)
mesh_velocities = csdl.Variable(value=mesh_velocities)
 
mesh_list = [mesh]
mesh_velocity_list = [mesh_velocities]
 
output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(mesh_list, mesh_velocity_list, dt=0.05, free_wake=False)
 
 
# mesh = mesh_dict['surface_0']['mesh'].value
coll_points = mesh_dict['surface_0']['panel_center']
Cp = output_dict['surface_0']['Cp']
CL  = output_dict['surface_0']['CL']
CDi = output_dict['surface_0']['CDi']
wake_mesh = wake_mesh_dict['surface_0']['mesh']
 
 
# CL = output_dict['surface_0']['CL'].value
 
# CL_norm = csdl.norm(output_dict['surface_0']['CL'])
 
# dCL_dmesh = csdl.derivative(CL_norm, mesh_velocities)
 
recorder.stop()
import jax
jax_gpu = jax.devices('gpu')[0]
name = jax_gpu.device_kind
stats = {}
for dtype in ['f64', 'f32']:
    # for device in ['cpu', 'gpu']:
    for device in ['gpu']:
        if dtype == 'f64':
            use_64 = True
        else:
            use_64 = False

        if device == 'gpu':
            use_gpu = True
            print(name)
        else:
            use_gpu = False
            name = device

        jax_sim = csdl.experimental.JaxSimulator(
            recorder=recorder,
            additional_inputs=[mesh, mesh_velocities], # list of outputs (put in csdl variable)
            additional_outputs=[mu, sigma, mu_wake, wake_mesh, coll_points, Cp, CL, CDi], # list of outputs (put in csdl variable)
            gpu=use_gpu,
            f64=use_64,
        )
        import time
        start = time.time()
        jax_sim.run()
        end = time.time()
        jax_sim.run()
        end1 = time.time()
        run_time = end1 - end
        compile_time = (end - start )- run_time

        print('Compile time: ', compile_time)
        print('Run time: ', run_time)
        
        # mesh = jax_sim[mesh]
        # coll_points = jax_sim[coll_points]
        # Cp = jax_sim[Cp]
        CLn = jax_sim[CL]
        # CDi = jax_sim[CDi]
        # mu = jax_sim[mu]
        # mu_value = mu[0,-2,:].reshape((nc-1)*2,ns-1)

        print(CLn)
        jax_sim.run()
        CLn2 = jax_sim[CL]

        stats[(dtype, device)] = {
            'nt': nt,
            'ns': ns,
            'nc': nc,
            'device_name': name,
            'CL': CLn,
            'CL (second run)': CLn2,
            'run_time': run_time,
            'compile_time': compile_time,
        }
import pickle
with open(f'{name}_benchmark_data.pickle', 'wb') as handle:
    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
# import matplotlib.pyplot as plt
# for key in stats.keys():
#     plt.plot(stats[key]['CL'][0], label=f'{key} - {stats[key]["run_time"]:.2f}s')
# plt.legend()
# plt.show()
exit()
