import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

ns = 11
nc = 3
b = 10
c = 1

wing_spacing = 2.*b

num_nodes = 1
alpha = np.array([5.]).reshape((num_nodes,))
V_inf = np.array([248.136, 0., 0.]).reshape((num_nodes, 3))

mesh = gen_vlm_mesh(ns, nc, b, c, frame='default')
new_mesh = np.zeros((num_nodes,) + mesh.shape)
for i in range(num_nodes):
    new_mesh[i,:,:,:] = mesh

mesh_dict = {}
mesh_dict['wing'] = new_mesh

recorder = csdl.Recorder(inline=True)
recorder.start()

ac_states_dummy  = 0.
output_dict = vlm_solver(mesh_dict, V_inf, alpha*np.pi/180.)
recorder.stop()

recorder.print_graph_structure()
recorder.visualize_graph(filename='ex2_mls_graph')

wing_CL = output_dict['wing']['CL'].value
wing_CDi = output_dict['wing']['CDi'].value

wing_CL_OAS = np.array([0.4426841725811703])
wing_CDi_OAS = np.array([0.005878842561184834])

CL_error = (wing_CL_OAS - wing_CL)/(wing_CL_OAS) * 100
CDi_error = (wing_CDi_OAS - wing_CDi)/(wing_CDi_OAS) * 100

print('======== ERROR PERCENTAGES (OAS - VortexAD) ========')
print(f'CL error (%): {CL_error}')
print(f'CDi error (%): {CDi_error}')