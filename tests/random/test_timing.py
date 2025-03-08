import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh
from VortexAD import steady_panel_solver

import time

'''
do 1000, 5000, 10000, 15000 panels
ns = 11
nc = 51, 251, 501, 751

modes:
1 -> up to system assembly
2 -> up to linear system solve
3 -> up to just after linear system solve
4 -> full panel method
'''
mode = 4
ns = 11
nc = 251


b = 2.
c = 0.2
num_nodes = 1

alpha_deg = 10.
alpha = np.deg2rad(alpha_deg) # aoa

mach = 0.15
sos = 340.3
Vx = sos*mach
V_inf = np.array([-Vx, 0., 0.])

mesh_orig = gen_panel_mesh(nc, ns, c, b, span_spacing='cosine',  frame='default', plot_mesh=False) # even chordwise spacing

mesh = np.zeros((num_nodes,) + mesh_orig.shape)
for i in range(num_nodes):
    mesh[i,:] = mesh_orig

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocities = np.zeros_like(mesh)
for i in range(num_nodes):
    mesh_velocities[i,:] = V_inf_rot

recorder = csdl.Recorder(inline=False)
start_graph = time.time()
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocities = csdl.Variable(value=mesh_velocities)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocities]


if mode == 1:
    sigma = steady_panel_solver(
        mesh_list, 
        mesh_velocity_list
    )
    asdf = csdl.average(sigma)
    outputs = [asdf]

elif mode == 2:
    AIC_mu = steady_panel_solver(
        mesh_list, 
        mesh_velocity_list
    )
    asdf = csdl.average(AIC_mu)
    outputs = [asdf]

elif mode == 3:
    mu = steady_panel_solver(
        mesh_list, 
        mesh_velocity_list
    )
    asdf = csdl.average(mu)
    outputs = [asdf]

elif mode == 4:
    output_dict, mesh_dict, mu, sigma = steady_panel_solver(
        mesh_list, 
        mesh_velocity_list
    )
    Cp = output_dict['surface_0']['Cp']
    asdf = csdl.average(Cp)
    outputs = [asdf]

recorder.stop()
end_graph = time.time()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh], # list of outputs (put in csdl variable)
    additional_outputs=outputs, # list of outputs (put in csdl variable)
)

jax_sim.run()

iterations = 10
start_total = time.time()
for i in range(iterations):
    print(f'iteration {i+1}')
    start_run = time.time()
    jax_sim.run()
    end_run = time.time()
    print(f'run time during iteration {i+1}: {end_run-start_run} seconds')
end_total = time.time()
fwd_eval_time = (end_total-start_total)/iterations
print(f'total time: {fwd_eval_time} seconds')
asdf = jax_sim[asdf]

