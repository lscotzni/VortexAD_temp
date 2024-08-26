import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
import matplotlib.pyplot as plt

from VortexAD.utils.plot import plot_wireframe, plot_pressure_distribution

import pyvista as pv

L = 10 # length
D = 1 # diameter

num_theta = 31
num_span = 41
nt = 20
theta = np.linspace(0., 2*np.pi, num_theta)
span_array = np.linspace(-L/2, L/2, num_span)

mesh = np.zeros((1, nt, num_theta, num_span, 3))
r = D/2
for i in range(num_span):
    mesh[0,:,:,i,0] = r*np.cos(theta)
    mesh[0,:,:,i,1] = span_array[i]
    mesh[0,:,:,i,2] = -r*np.sin(theta)

# plt.figure()
# plt.plot(mesh[0,0,:,0,0], mesh[0,0,:,0,2])
# plt.show()
# exit()
    
V_inf = np.array([-100., 0., 0.])
alpha_deg = 0.
alpha = np.deg2rad(alpha_deg) # aoa
V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocities = np.zeros_like(mesh)

for j in range(nt):
    mesh_velocities[:,j,:] = V_inf

recorder = csdl.Recorder(inline=False)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocities = csdl.Variable(value=mesh_velocities)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocities]

output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(mesh_list, mesh_velocity_list, dt=0.01, free_wake=False)

coll_points = mesh_dict['surface_0']['panel_center']
Cp = output_dict['surface_0']['Cp']
CL  = output_dict['surface_0']['CL']
CDi = output_dict['surface_0']['CDi']
wake_mesh = wake_mesh_dict['surface_0']['mesh']

recorder.stop()
print('stopping recorder')
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocities], # list of outputs (put in csdl variable)
    additional_outputs=[mu, sigma, mu_wake, wake_mesh, coll_points, Cp, CL, CDi], # list of outputs (put in csdl variable)
    derivatives_kwargs={'concatenate_ofs': True}
)
print('running jax simulator')
jax_sim.run()


mesh = jax_sim[mesh]
coll_points = jax_sim[coll_points]
Cp = jax_sim[Cp]
CL = jax_sim[CL]
CDi = jax_sim[CDi]
mu = jax_sim[mu]
mu_wake = jax_sim[mu_wake]
wake_mesh = jax_sim[wake_mesh]

if False:
    plot_pressure_distribution(mesh, Cp, interactive=True, top_view=False)

if True:
    # plot_wireframe(mesh, wake_mesh, mu.value, mu_wake.value, nt, interactive=False, backend='cv', name=f'wing_fw_{alpha_deg}')
    plot_wireframe(mesh, wake_mesh, mu, mu_wake, nt, interactive=False, backend='cv', name='cylinder_pw_side', side_view=False)


Cp_analytical = 1 - 4*np.sin((theta[:-1] + theta[1:])/2)**2
Cp_csdl = Cp[0,-2,:,int((num_span-1)/2)]

plt.figure()
plt.plot((theta[:-1] + theta[1:])/2*180/np.pi, Cp_analytical, 'k-', label='Analytical')
plt.plot((theta[:-1] + theta[1:])/2*180/np.pi, Cp_csdl, 'rv', label='CSDL')
plt.xlabel('Angle (degrees)')
plt.ylabel('Cp')
plt.grid()
plt.legend()
plt.show()
