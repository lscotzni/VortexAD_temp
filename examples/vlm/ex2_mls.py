import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

ns = 11
nc = 5
b = 10
c = 1

wing_spacing = 0.5*b

num_nodes = 1
alpha = np.array([25.]).reshape((num_nodes,)) * np.pi/180.
V_inf = np.array([60., 0., 0.])

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_orig = gen_vlm_mesh(ns, nc, b, c, frame='caddee')
mesh_2_orig = gen_vlm_mesh(ns, nc, b, c, frame='caddee')
mesh_2_orig[:,:,0] -= wing_spacing

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

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocity]

# mesh_list = [mesh, mesh_2]
# mesh_velocity_list = [mesh_velocity, mesh_velocity_2]

recorder = csdl.Recorder(inline=True)
recorder.start()
output_vg = vlm_solver(mesh_list, mesh_velocity_list)
recorder.stop()

# recorder.print_graph_structure()
# recorder.visualize_graph(filename='ex2_mls_graph')

print('======  PRINTING TOTAL OUTPUTS ======')
print('Total force (N): ', output_vg.total_force.value)
print('Total Moment (Nm): ', output_vg.total_moment.value)
print('Total lift (N): ', output_vg.total_lift.value)
print('Total drag (N): ', output_vg.total_drag.value)

print('======  PRINTING OUTPUTS PER SURFACE ======')
for i in range(len(mesh_list)): # LOOPING THROUGH NUMBER OF SURFACES
    print('======  SURFACE 1 ======')
    print('Surface total force (N): ', output_vg.surface_force[i].value)
    print('Surface total moment (Nm): ', output_vg.surface_moment[i].value)
    print('Surface total lift (N): ', output_vg.surface_lift[i].value)
    print('Surface total drag (N): ', output_vg.surface_drag[i].value)
    print('Surface CL: ', output_vg.surface_CL[i].value)
    print('Surface CDi : ', output_vg.surface_CDi[i].value)

    # print('Surface panel forces (N): ', output_vg.surface_panel_forces[i].value)
    # print('Surface sectional center of pressure (m): ', output_vg.surface_sectional_cop[i].value)
    # print('Surface total center of pressure (m): ', output_vg.surface_cop[i].value)