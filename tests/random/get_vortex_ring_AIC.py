import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.vlm.vlm_solver import vlm_solver
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

from VortexAD import SAMPLE_GEOMETRY_PATH

import pyvista as pv

alpha_deg = 0.
alpha = np.deg2rad(alpha_deg) # aoa

mach = 0.3
sos = 340.3
# V_inf = np.array([sos*mach, 0., 0.])
V_inf = np.array([-10., 0., 0.])
nt = 10
num_nodes = 1

ns, nc = 5, 11

filename = str(SAMPLE_GEOMETRY_PATH) + '/pm/wing_NACA0012_ar10.vtk' 
mesh_data = pv.read(filename)
mesh_orig = mesh_data.points.reshape((21,5,3))

upper_surf = mesh_orig[(nc-1):,:,:]

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_steady = np.zeros((num_nodes,) + upper_surf.shape)
mesh_steady_velocities = np.zeros_like(mesh_steady)
mesh_unsteady = np.zeros((num_nodes, nt) + upper_surf.shape)
mesh_unsteady_velocities = np.zeros_like(mesh_unsteady)
for i in range(num_nodes):
    mesh_steady[i,:] = upper_surf
    mesh_steady_velocities[i,:] = V_inf_rot
    for j in range(nt):
        mesh_unsteady[i,j,:] = upper_surf
        mesh_unsteady_velocities[i,j,:] = V_inf_rot

recorder = csdl.Recorder(inline=True)
recorder.start()

mesh_steady = csdl.Variable(value=mesh_steady)
mesh_unsteady = csdl.Variable(value=mesh_unsteady)
mesh_steady_velocities = csdl.Variable(value=mesh_steady_velocities)
mesh_unsteady_velocities = csdl.Variable(value=mesh_unsteady_velocities)

vlm_mesh_list = [mesh_steady]
vlm_mesh_velocity_list = [mesh_steady_velocities]

panel_solver_mesh_list = [mesh_unsteady]
panel_solver_mesh_velocity_list = [mesh_unsteady_velocities]

output_vg_vlm = vlm_solver(vlm_mesh_list, vlm_mesh_velocity_list)

output_dict_pm, mesh_dict_pm, wake_mesh_dict_pm, gamma_pm, gamma_wake = unsteady_panel_solver(
    mesh_list=panel_solver_mesh_list, 
    mesh_velocity_list=panel_solver_mesh_velocity_list, 
    dt=0.05, 
    mode='vortex-ring', 
    free_wake=False
)