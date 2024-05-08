import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

ns = 11
nc = 5
b = 10
c = 1

wing_spacing = 10.*b

num_nodes = 1
alpha = np.array([5.]).reshape((num_nodes,))
V_inf = np.array([-60., 0., 0.]).reshape((num_nodes, 3))

mesh = gen_vlm_mesh(ns, nc, b, c, frame='caddee')
mesh_2 = gen_vlm_mesh(ns+1, nc+1, b, c, frame='caddee')
mesh_2[:,:,1] += wing_spacing

new_mesh = np.zeros((num_nodes,) + mesh.shape)
new_mesh_2 = np.zeros((num_nodes,) + mesh_2.shape)

for i in range(num_nodes):
    new_mesh[i,:,:,:] = mesh
    new_mesh_2[i,:,:,:] = mesh_2

mesh_dict = {}
mesh_dict['wing'] = new_mesh
mesh_dict['wing_2'] = new_mesh_2

recorder = csdl.Recorder(inline=True)
recorder.start()

ac_states_dummy  = 0.
output_dict = vlm_solver(mesh_dict, ac_states_dummy, V_inf, alpha*np.pi/180.)
recorder.stop()

wing_CL = output_dict['wing']['CL'].value
wing_CL_2 = output_dict['wing_2']['CL'].value