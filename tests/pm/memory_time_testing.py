import numpy as np 
import csdl_alpha as csdl 

from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh
import time
# from memory_profiler import profile
# @profile

from guppy import hpy 

h=hpy()

def grid_indep():

    b = 10
    c = 1.564
    ns = 41
    nc = 26

    alpha = np.deg2rad(0.) # aoa

    mach = 0.25
    sos = 340.3
    V_inf = np.array([sos*mach, 0., 0.])
    nt = 5
    num_nodes = 1

    mesh_orig = gen_panel_mesh(nc, ns, c, b, frame='default', plot_mesh=False)

    mesh = np.zeros((num_nodes, nt) + mesh_orig.shape)
    for i in range(num_nodes):
        for j in range(nt):
            mesh[i,j,:] = mesh_orig

    V_rot_mat = np.zeros((3,3))
    V_rot_mat[1,1] = 1.
    V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
    V_rot_mat[2,0] = np.sin(alpha)
    V_rot_mat[0,2] = -np.sin(alpha)
    V_inf_rot = np.matmul(V_rot_mat, V_inf)

    mesh_velocities = np.zeros_like(mesh)
    for i in range(num_nodes):
        for j in range(nt):
            mesh_velocities[i,j,:] = V_inf_rot

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    mesh = csdl.Variable(value=mesh)
    mesh_velocities = csdl.Variable(value=mesh_velocities)

    mesh_list = [mesh]
    mesh_velocity_list = [mesh_velocities]

    outputs = unsteady_panel_solver(mesh_list, mesh_velocity_list, dt=0.01)
    recorder.stop()

if __name__ == '__main__':
    start = time.time()
    grid_indep()
    stop = time.time()

    print(f'Run time: {stop - start} seconds')

    heap = h.heap()
    print(heap)
