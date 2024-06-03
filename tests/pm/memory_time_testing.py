import numpy as np 
import csdl_alpha as csdl 

from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh
import time
# from memory_profiler import profile
# @profile

from guppy import hpy 

h=hpy()

def grid_indep(nc, ns, nt):

    b = 10
    c = 1.564
    # ns = 41
    # nc = 26

    alpha = np.deg2rad(10.) # aoa

    mach = 0.25
    sos = 340.3
    V_inf = np.array([sos*mach, 0., 0.])
    # nt = 5
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

    start_time = time.time()

    output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(mesh_list, mesh_velocity_list, dt=0.1, free_wake=True)
    run_time = time.time()

    CL = output_dict['surface_0']['CL']

    # dCL_dmesh = csdl.derivative(csdl.norm(CL),mesh)
    deriv_time = time.time()
    recorder.stop()

    print(f'Run time: {run_time - start_time} seconds')
    # print(f'Derivative time: {deriv_time - run_time} seconds')

    return recorder

if __name__ == '__main__':
    nt = 8
    # ns_dict = {
    #     5: {'nc': [5, 11, 21, 31, 41], 't_forward': {[]}, 't_deriv': {[]}},
    #     11:{'nc': [5, 11, 21, 31, 41], 't_forward': {[]}, 't_deriv': {[]}},
    #     21:{'nc': [5, 11, 21, 31, 41], 't_forward': {[]}, 't_deriv': {[]}},
    #     31:{'nc': [5, 11, 21, 26, 31, 36], 't_forward': {[]}, 't_deriv': {[]}},
    #     41:{'nc': [5, 11, 21, 26], 't_forward': {[]}, 't_deriv': {[]}},
    # }

    recorder = grid_indep(nc=11,ns=15,nt=nt)

    heap = h.heap()
    print(heap)
