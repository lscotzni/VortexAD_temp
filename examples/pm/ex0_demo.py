import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh, gen_panel_mesh_new
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

# plotting functions
from VortexAD.utils.plot import plot_wireframe, plot_pressure_distribution

b = 10.
c = 1.
ns = 15
nc = 21
dt = 0.05
nt = 30
num_nodes = 1

alpha_deg = 10.
alpha = np.deg2rad(alpha_deg) # aoa

mach = 0.15
sos = 340.3
Vx = sos*mach
V_inf = np.array([-Vx, 0., 0.])

mesh_orig = gen_panel_mesh(nc, ns, c, b, span_spacing='cosine',  frame='default', plot_mesh=False) # even chordwise spacing
# mesh_orig = gen_panel_mesh_new(nc, ns, c, b,  frame='default', plot_mesh=False) # uneven chordwise spacing

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

'''
inputs to the solver:
- list of meshes
- list of mesh velocities
- actuation velocities (if rotating bodies) -> not applicable for now
'''
output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(
    mesh_list, 
    mesh_velocity_list, 
    dt=dt, 
    free_wake=True
)

coll_points = mesh_dict['surface_0']['panel_center']
Cp = output_dict['surface_0']['Cp']
CL  = output_dict['surface_0']['CL']
CDi = output_dict['surface_0']['CDi']
wake_mesh = wake_mesh_dict['surface_0']['mesh']

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocities], # list of outputs (put in csdl variable)
    additional_outputs=[mu, sigma, mu_wake, wake_mesh, coll_points, Cp, CL, CDi], # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh = jax_sim[mesh]
coll_points = jax_sim[coll_points]
Cp = jax_sim[Cp]
CL = jax_sim[CL]
CDi = jax_sim[CDi]
mu = jax_sim[mu]
mu_wake = jax_sim[mu_wake]
wake_mesh = jax_sim[wake_mesh]

mu_value = mu[0,-2,:].reshape((nc-1)*2,ns-1)

print('doublet distribution:')
print(mu_value)
print(f'CL: {CL}')
print(f'CDi: {CDi}')

if True:
    plot_pressure_distribution(mesh, Cp, interactive=True, top_view=False)

if True:
    plot_wireframe([mesh], [wake_mesh], [mu], [mu_wake], nt, interactive=False, backend='cv', name='demo_animation')
