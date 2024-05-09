import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

ns = 11
nc = 3
b = 10
c = 1

num_nodes = 1
alpha = np.array([5.]).reshape((num_nodes,)) * np.pi/180.
V_inf = np.array([248.136, 0., 0.])

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_orig = gen_vlm_mesh(ns, nc, b, c, frame='default')
mesh = np.zeros((num_nodes,) + mesh_orig.shape)
for i in range(num_nodes):
    mesh[i,:,:,:] = mesh_orig

mesh_velocity = np.zeros_like(mesh)
mesh_velocity[:,:,:,0] = V_inf_rot[0]
mesh_velocity[:,:,:,2] = V_inf_rot[2]

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocity]

recorder = csdl.Recorder(inline=True)
recorder.start()
output_vg = vlm_solver(mesh_list, mesh_velocity_list)
recorder.stop()

# recorder.print_graph_structure()
# recorder.visualize_graph(filename='ex2_mls_graph')

wing_CL = output_vg.surface_CL.value
wing_CDi = output_vg.surface_CDi.value

wing_CL_OAS = np.array([0.4426841725811703])
wing_CDi_OAS = np.array([0.005878842561184834])

CL_error = (wing_CL_OAS - wing_CL)/(wing_CL_OAS) * 100
CDi_error = (wing_CDi_OAS - wing_CDi)/(wing_CDi_OAS) * 100

print('======== ERROR PERCENTAGES (OAS - VortexAD) ========')
print(f'CL error (%): {CL_error}')
print(f'CDi error (%): {CDi_error}')


1