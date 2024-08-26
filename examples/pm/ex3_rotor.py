import numpy as np
import csdl_alpha as csdl
 
from VortexAD.core.geometry.gen_rotor_mesh import gen_rotor_mesh
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
 
from VortexAD.utils.plot import plot_wireframe, plot_pressure_distribution
 
radius = 0.75
r_inner = 0.25
max_chord = 0.1
 
chord_array = np.array([0.55, 0.65, 0.75, 0.85, 0.9, 1.0, 1.0, 1.0, 0.95]) * max_chord
ns = len(chord_array)
span_array = np.linspace(0.25, 1., ns) * radius
nc = 7
blade_1_mesh, blade_2_mesh = gen_rotor_mesh(nc, chord_array, radius, r_inner, twist=-15., plot_mesh=False)
 
rpm = 1000
omega = rpm*2*np.pi/60.
omega_vec = np.array([0.,0.,omega])
nt = 40
dt = 0.005
free_stream = np.array([0., 0., 5])
 
rot_mat = np.zeros((nt, 3, 3))
for i in range(nt):
    angle = omega_vec[2]*dt*i
    rot_mat[i,2,2] = 1
    rot_mat[i,0,0] = np.cos(angle)
    rot_mat[i,1,1] = np.cos(angle)
    rot_mat[i,0,1] = np.sin(angle)
    rot_mat[i,1,0] = -np.sin(angle)
 
blade_1_mesh_vel = np.cross(omega_vec, blade_1_mesh, axisa=0, axisb=2)
blade_2_mesh_vel = np.cross(omega_vec, blade_2_mesh, axisa=0, axisb=2)
 
blade_1_coll_point = (blade_1_mesh[:-1,:-1,:]+blade_1_mesh[1:,:-1,:]+blade_1_mesh[1:,1:,:]+blade_1_mesh[:-1,1:,:])/4.
blade_2_coll_point = (blade_2_mesh[:-1,:-1,:]+blade_2_mesh[1:,:-1,:]+blade_2_mesh[1:,1:,:]+blade_2_mesh[:-1,1:,:])/4.
blade_1_coll_vel = np.cross(omega_vec, blade_1_coll_point, axisa=0, axisb=2)
blade_2_coll_vel = np.cross(omega_vec, blade_2_coll_point, axisa=0, axisb=2)
 
blade_1_mesh_exp = np.zeros((1, nt) + blade_1_mesh.shape)
blade_2_mesh_exp = np.zeros((1, nt) + blade_2_mesh.shape)
 
blade_1_mesh_vel_exp = np.zeros((1, nt) + blade_1_mesh.shape)
blade_2_mesh_vel_exp = np.zeros((1, nt) + blade_2_mesh.shape)
 
 
for i in range(nt):
    blade_1_mesh_exp[0,i,:,:,:] = np.einsum('ij,abi->abj', rot_mat[i,:,:], blade_1_mesh)
    blade_2_mesh_exp[0,i,:,:,:] = np.einsum('ij,abi->abj', rot_mat[i,:,:], blade_2_mesh)
 
    blade_1_mesh_vel_exp[0,i,:,:,2] = free_stream[2]
    blade_2_mesh_vel_exp[0,i,:,:,2] = free_stream[2]
 
blade_1_coll_vel_exp = np.einsum('ijk,ab->abijk', blade_1_coll_vel, np.ones((1, nt)))
blade_2_coll_vel_exp = np.einsum('ijk,ab->abijk', blade_2_coll_vel, np.ones((1, nt)))
 
recorder = csdl.Recorder(inline=False)
recorder.start()
 
blade_1_mesh_exp = csdl.Variable(value=blade_1_mesh_exp)
blade_2_mesh_exp = csdl.Variable(value=blade_2_mesh_exp)
blade_1_mesh_vel_exp = csdl.Variable(value=blade_1_mesh_vel_exp)
blade_2_mesh_vel_exp = csdl.Variable(value=blade_2_mesh_vel_exp)
blade_1_coll_vel_exp = csdl.Variable(value=blade_1_coll_vel_exp)
blade_2_coll_vel_exp = csdl.Variable(value=blade_2_coll_vel_exp)
 
# mesh_list = [blade_1_mesh_exp, blade_2_mesh_exp]
# mesh_vel_list = [blade_1_mesh_vel_exp, blade_2_mesh_vel_exp]
# coll_vel_list = [blade_1_coll_vel_exp, blade_2_coll_vel_exp]
 
mesh_list = [blade_1_mesh_exp]
mesh_vel_list = [blade_1_mesh_vel_exp]
coll_vel_list = [blade_1_coll_vel_exp]
 
output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(mesh_list, mesh_vel_list, coll_vel_list, dt=dt, free_wake=False)
 
blade_1_mesh = mesh_dict['surface_0']['mesh']
blade_1_wake = wake_mesh_dict['surface_0']['mesh']
# blade_2_mesh = mesh_dict['surface_1']['mesh'].value
# blade_2_wake = wake_mesh_dict['surface_2']['mesh'].value

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[blade_1_mesh_exp], # list of outputs (put in csdl variable)
    additional_outputs=[mu, sigma, mu_wake, blade_1_wake], # list of outputs (put in csdl variable)
)
jax_sim.run()

blade_1_mesh = jax_sim[blade_1_mesh_exp]
mu = jax_sim[mu]
mu_wake = jax_sim[mu_wake]
blade_1_wake = jax_sim[blade_1_wake]
 
if True:
    plot_wireframe(blade_1_mesh, blade_1_wake, mu, mu_wake, nt, interactive=False, backend='cv', name='free_wake_demo')
