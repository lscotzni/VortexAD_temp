import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

b = 10
c = 1
ns = 3
nc = 3

alpha = 0. # aoa

mach = 0.25
sos = 340.3
V_inf = np.array([sos*mach, 0., 0.])
nt = 2
num_nodes = 1

mesh_orig = gen_panel_mesh(nc, ns, c, b, frame='caddee', plot_mesh=False)

mesh = np.zeros((num_nodes, nt) + mesh_orig.shape)
for i in range(num_nodes):
    for j in range(nt):
        mesh[i,j,:] = mesh_orig

mesh_velocities = np.zeros_like(mesh)
for i in range(num_nodes):
    for j in range(nt):
        mesh_velocities[i,j,:] = V_inf * -1.

recorder = csdl.Recorder(inline=True)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocities = csdl.Variable(value=mesh_velocities)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocities]

mesh_dict = unsteady_panel_solver(mesh_list, mesh_velocity_list)
recorder.stop()

