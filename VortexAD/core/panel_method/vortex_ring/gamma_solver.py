import csdl_alpha as csdl 

from VortexAD.core.panel_method.vortex_ring.initialize_unsteady_wake import initialize_unsteady_wake
from VortexAD.core.panel_method.vortex_ring.transient_solver import transient_solver

def gamma_solver(num_nodes, nt, mesh_dict, dt, free_wake=False):

    surface_names = list(mesh_dict.keys())
    num_tot_panels = 0
    for surface in surface_names:
        num_tot_panels += mesh_dict[surface]['num_panels']

    wake_mesh_dict = initialize_unsteady_wake(mesh_dict, num_nodes, dt, panel_fraction=100.)

    gamma, gamma_wake = transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=free_wake)

    return gamma, gamma_wake, wake_mesh_dict