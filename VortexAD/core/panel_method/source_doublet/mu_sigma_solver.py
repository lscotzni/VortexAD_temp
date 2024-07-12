import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.source_doublet.initialize_unsteady_wake import initialize_unsteady_wake
from VortexAD.core.panel_method.source_doublet.transient_solver import transient_solver
from VortexAD.core.panel_method.source_doublet.transient_solver_new import transient_solver_new

def mu_sigma_solver(num_nodes, nt, mesh_dict, dt, free_wake=False):

    surface_names = list(mesh_dict.keys())
    num_tot_panels = 0
    for surface in surface_names:
        num_tot_panels += mesh_dict[surface]['num_panels']

    wake_mesh_dict = initialize_unsteady_wake(mesh_dict, num_nodes, dt, panel_fraction=1.)

    # sigma = compute_source_strengths(mesh_dict, surface_names, num_nodes, nt, num_tot_panels) # shape=(num_nodes, nt, num_surf_panels)

    if free_wake:
        mu, sigma, mu_wake = transient_solver_new(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=free_wake)
    
        # return mu, sigma, wake_mesh_dict, induced_vel
        return mu, sigma, mu_wake, wake_mesh_dict
    else:
        mu, sigma, mu_wake = transient_solver_new(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt)
    
        return mu, sigma, mu_wake, wake_mesh_dict

    






'''
Structure of unsteady panel solver
- initialize wake model
    - add the first set of wake panels propagated with the free-stream to about 0.2-0.3 of the distance it should travel back
- compute source strengths (& corresponding AIC)
    - doesn't need time-stepping because we have the free-stream velocity and the time-dependent mesh
- time-stepping solver
    - compute doublet AIC
    - compute kutta condition influence on AIC
    - propagate wake
        - FREE WAKE: RECOMPUTE AIC FOR WAKES AND COMPUTE INDUCED VELOCITIES
    - move to next time step

- return mu, sigma

'''