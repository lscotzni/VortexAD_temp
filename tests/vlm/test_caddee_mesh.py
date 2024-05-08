import numpy as np
import csdl_alpha as csdl 
from VortexAD.core.vlm.vlm_solver import vlm_solver
from VortexAD import SAMPLE_GEOMETRY_PATH

file_name = str(SAMPLE_GEOMETRY_PATH) + '/vlm_mesh_linear_spacing.npy'

mesh = np.load(file_name)

num_nodes = 1
new_mesh = np.zeros((num_nodes,) + mesh.shape)
for i in range(num_nodes):
    new_mesh[i,:,:,:] = mesh

mesh_dict = {'wing': mesh}
exit()

alpha = 5. 

recorder = csdl.Recorder(inline=True)
recorder.start()

vlm_solver(mesh_dict, alpha)