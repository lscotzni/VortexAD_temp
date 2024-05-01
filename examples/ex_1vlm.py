import numpy as np
import csdl_alpha as csdl 

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

ns = 5
nc = 3
b = 10
c = 1

num_nodes = 1

alpha = 5 # angle of attack in degrees

mesh = gen_vlm_mesh(ns, nc, b, c)
new_mesh = np.zeros((num_nodes,) + mesh.shape)

mesh_2 = gen_vlm_mesh(ns+1, nc+1, b+1, c+1)
new_mesh_2 = np.zeros((num_nodes,) + mesh_2.shape)

for i in range(num_nodes):
    new_mesh[i,:,:,:] = mesh
    new_mesh_2[i,:,:,:] = mesh_2
mesh_dict = {}
mesh_dict['wing'] = new_mesh # (num_nodes, nc, ns, 3)
mesh_dict['wing_2'] = new_mesh_2

recorder = csdl.Recorder(inline=True)
recorder.start()

vlm_solver(mesh_dict, alpha)
recorder.stop()