import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

ns = 11
nc = 5
b = 10
c = 1

wing_spacing = 2.*b

num_nodes = 1
alpha = np.array([5.]).reshape((num_nodes,)) * np.pi/180.
V_inf = np.array([-60., 0., 0.])

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_orig = gen_vlm_mesh(ns, nc, b, c, frame='caddee')
mesh_2_orig = gen_vlm_mesh(ns, nc, b, c, frame='caddee')
mesh_2_orig[:,:,1] += wing_spacing

mesh = np.zeros((num_nodes,) + mesh_orig.shape)
mesh_2 = np.zeros((num_nodes,) + mesh_2_orig.shape)

for i in range(num_nodes):
    mesh[i,:,:,:] = mesh_orig
    mesh_2[i,:,:,:] = mesh_2_orig

mesh_velocity = np.zeros_like(mesh)
mesh_velocity[:,:,:,0] = V_inf_rot[0]
mesh_velocity[:,:,:,2] = V_inf_rot[2]


mesh_velocity_2 = np.zeros_like(mesh_2)
mesh_velocity_2[:,:,:,0] = V_inf_rot[0]
mesh_velocity_2[:,:,:,2] = V_inf_rot[2]

mesh_list = [mesh, mesh_2]
mesh_velocity_list = [mesh_velocity, mesh_velocity_2]

recorder = csdl.Recorder(inline=True)
recorder.start()
output_vg = vlm_solver(mesh_list, mesh_velocity_list)
recorder.stop()

# recorder.print_graph_structure()
# recorder.visualize_graph(filename='ex2_mls_graph')

wing_CL = output_vg.surface_CL.value[0,0]
wing_CL_2 = output_vg.surface_CL.value[0,1]

print('======== CL OF LIFTING SURFACES ========')
print(f'CL OF SURFACE 1: {wing_CL}')
print(f'CL OF SURFACE 2: {wing_CL_2}')


1