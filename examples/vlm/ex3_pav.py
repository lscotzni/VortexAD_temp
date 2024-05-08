import numpy as np
import matplotlib.pyplot as plt
import csdl_alpha as csdl 

from VortexAD import SAMPLE_GEOMETRY_PATH
from VortexAD.core.vlm.vlm_solver import vlm_solver

# ====== GEOMETRY SETUP ======
pav_mesh_path = str(SAMPLE_GEOMETRY_PATH) + '/PAV_mesh.txt'
pav_mesh_data = np.genfromtxt(pav_mesh_path)
nc, ns = 20, 16

pav_half_mesh = np.swapaxes(pav_mesh_data.reshape((20, 16, 3)), 0,1)
pav_other_half_mesh = pav_half_mesh[:,::-1,:].copy() # reverse the ordering
pav_other_half_mesh[:,:,1] *= -1.

pav_mesh = np.concatenate((pav_other_half_mesh, pav_half_mesh), axis=1)
nc, ns = 16, 40

alpha = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]) # pitch in degrees
# alpha = np.array([0., 4.]) # pitch in degrees
alpha_rad = np.deg2rad(alpha)

num_nodes = alpha.shape[0]
pav_new_mesh = np.zeros((num_nodes,) + pav_mesh.shape)
for i in range(num_nodes):
    pav_new_mesh[i,:,:,:] = pav_mesh

mesh_dict = {}
mesh_dict['pav'] = pav_new_mesh
v_mag = 248.136
V_inf = np.zeros((num_nodes,3))
V_inf[:,0] = 248.136

recorder = csdl.Recorder(inline=True)
recorder.start()

ac_states_dummy  = 0.
output_dict = vlm_solver(mesh_dict, V_inf, alpha_rad)
recorder.stop()
recorder.print_graph_structure()
recorder.visualize_graph(filename='ex3_pav_graph')

wing_CL = output_dict['pav']['CL'].value

wing_CL_AVL = np.array([-0.77020, -0.61878, -0.465579217135667, -0.311163208143354,-0.155784316259393,0.00000,
                          0.155784316259393, 0.311163208143354, 0.465579217135667, 0.61878, 0.77020])

CL_error = (wing_CL_AVL - wing_CL) / (wing_CL_AVL + 1.e-12) * 100.

plt.figure()
plt.plot(alpha, CL_error)
plt.xlabel('angle of attack (deg)')
plt.ylabel('CL error (AVL - VortexAD) (%)')

plt.grid()
plt.show