import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

# flow parameters
frame = 'default'
vnv_scaler =  1.
num_nodes = 1
alpha_deg = 5.
alpha = np.array([alpha_deg,]) * np.pi/180.
V_inf = np.array([-60, 0., 0.])
if frame == 'caddee':
    V_inf *= -1.
    vnv_scaler = -1.

# grid setup
ns = 21
nc = 11
b = 10
c = 1
# nc, ns = 11, 15

# generating mesh
mesh_orig = gen_vlm_mesh(ns, nc, b, c, frame=frame)
mesh = np.zeros((num_nodes,) + mesh_orig.shape)
for i in range(num_nodes):
    mesh[i,:,:,:] = mesh_orig

# setting up mesh velocity
V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocity = np.zeros_like(mesh)
mesh_velocity[:,:,:,0] = V_inf_rot[0]
mesh_velocity[:,:,:,2] = V_inf_rot[2]

# solver input setup
recorder = csdl.Recorder(inline=False)
recorder.start()
mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_velocity)
mesh_list = [mesh]
mesh_velocity_list = [mesh_velocity]

output_vg = vlm_solver(mesh_list, mesh_velocity_list)
wing_CL = output_vg.surface_CL[0]
wing_CDi = output_vg.surface_CDi[0]
gamma = output_vg.gamma

mesh_dict = output_vg.mesh_dict
bound_vortex_mesh = mesh_dict['surface_0']['bound_vortex_mesh']
wake_vortex_mesh = mesh_dict['surface_0']['wake_vortex_mesh']
recorder.stop()



jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocity],
    additional_outputs = [wing_CL, wing_CDi, gamma, bound_vortex_mesh, wake_vortex_mesh]
)
jax_sim.run()
mesh = jax_sim[mesh]
mesh_velocity = jax_sim[mesh_velocity]
gamma = jax_sim[gamma]
wing_CL = jax_sim[wing_CL]
wing_CDi = jax_sim[wing_CDi]

bound_vortex_mesh = jax_sim[bound_vortex_mesh]
wake_vortex_mesh = jax_sim[wake_vortex_mesh]

from VortexAD.utils.plot import plot_pressure_distribution
gamma_grid = gamma.reshape((1, nc-1, ns-1))
gamma_horseshoe = np.zeros_like(gamma_grid)
gamma_horseshoe[:,0,:] = gamma_grid[:,0,:]
gamma_horseshoe[:,1:,:] = gamma_grid[:,1:,:] - gamma_grid[:,:-1,:]
plot_pressure_distribution([mesh], [gamma_grid], interactive=True, top_view=False)
# plot_pressure_distribution([mesh], [gamma_horseshoe], interactive=True, top_view=False)

data_mesh_dict = {
    'surface_0': {
        'mesh': mesh,
        'bound_vortex_mesh': bound_vortex_mesh,
        'wake_vortex_mesh': wake_vortex_mesh,
        'nc': nc,
        'ns': ns
    }
}

data = {
    'mesh_dict': data_mesh_dict,
    'velocity': mesh_velocity,
    'alpha': alpha_deg,
    'gamma': gamma,
}

import pickle
with open('superposition_data.pickle', 'wb') as file:
    pickle.dump(data, file)

