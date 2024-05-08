import numpy as np
import csdl_alpha as csdl 
import matplotlib.pyplot as plt

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

b = 100
c = 1

num_nodes = 1
frame = 'caddee'

alpha = 5. # angle of attack in degrees
V_inf = np.array([10., 0., 0.])
if frame == 'caddee':
    V_inf *= -1.

ns_list = np.array([3, 5, 15, 25, 45, 55])
nc_list = np.array([3, 5, 15])
# ns_list = np.array([3, 5, 10])
# nc_list = np.array([3, 5, 10])

num_ns = len(ns_list)
num_nc = len(nc_list)

CL  = np.zeros((num_ns, num_nc))
CDi = np.zeros_like(CL)

for i in range(num_ns):
    ns = ns_list[i]
    print(f'======= i = {i}, ns = {ns} =======')
    
    for j in range(num_nc):
        nc = nc_list[j]
        print(f'j = {j}, nc = {nc}')

        mesh = gen_vlm_mesh(ns, nc, b, c, frame=frame)
        new_mesh = np.zeros((num_nodes,) + mesh.shape)

        for k in range(num_nodes):
            new_mesh[k,:,:,:] = mesh
        mesh_dict = {}
        mesh_dict['wing'] = new_mesh # (num_nodes, nc, ns, 3)

        recorder = csdl.Recorder(inline=True)
        recorder.start()

        ac_states_dummy  = 0.
        output_dict = vlm_solver(mesh_dict, ac_states_dummy, V_inf, alpha*np.pi/180.)
        recorder.stop()

        CL[i,j] = output_dict['wing']['CL'].value
        CDi[i,j] = output_dict['wing']['CDi'].value

        del mesh, new_mesh, mesh_dict, output_dict, recorder

theory_value = 2*np.pi*np.sin(alpha*np.pi/180.)
if frame == 'caddee':
    theory_value *= -1.
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.plot([nc_list[0], nc_list[-1]], [theory_value]*2, 'k-*')
for i in range(num_ns):
    ax1.plot(nc_list, CL[i,:], label=f'ns = {ns_list[i]}')
    ax2.plot(nc_list, CDi[i,:], label=f'ns = {ns_list[i]}')

ax1.set_ylabel('CL')
ax1.grid(True)

ax2.set_ylabel('CDi')
ax2.set_xlabel('nc')
ax2.grid(True)


plt.legend()
plt.show()