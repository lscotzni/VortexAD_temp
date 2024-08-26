import numpy as np 

from VortexAD.core.panel_method.source_doublet.initialize_unsteady_wake import initialize_unsteady_wake
# from VortexAD.core.panel_method.source_doublet.initialize_unsteady_wake_new import initialize_unsteady_wake


from VortexAD.core.panel_method.source_doublet.transient_solver import transient_solver
# from VortexAD.core.panel_method.source_doublet.transient_solver_new import transient_solver
# from VortexAD.core.panel_method.source_doublet.transient_solver_noKC import transient_solver # version with no AIC adjustment

from VortexAD.core.panel_method.source_doublet.unstructured_transient_solver import unstructured_transient_solver

def mu_sigma_solver(num_nodes, nt, mesh_dict, dt, mesh_mode='structured', free_wake=False):

    if mesh_mode == 'structured':
        surface_names = list(mesh_dict.keys())
        num_tot_panels = 0
        for surface in surface_names:
            num_tot_panels += mesh_dict[surface]['num_panels']
    elif mesh_mode == 'unstructured':
        num_tot_panels = len(mesh_dict['cell_adjacency'])

    wake_mesh_dict = initialize_unsteady_wake(mesh_dict, num_nodes, dt, mesh_mode=mesh_mode, panel_fraction=0.25)

    if mesh_mode == 'structured':
        mu, sigma, mu_wake = transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=free_wake)
        # mu, sigma, mu_wake = transient_solver_new(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=free_wake)

    elif mesh_mode == 'unstructured':
        mu, sigma, mu_wake = unstructured_transient_solver(
            mesh_dict,
            wake_mesh_dict,
            num_nodes,
            nt,
            num_tot_panels,
            dt,
            free_wake=free_wake
        )

    return mu, sigma, mu_wake, wake_mesh_dict

    # OLD SETUP
    # note that the prescribed wake solver was working previously, but not the free-wake mode
    if free_wake:
        mu, sigma, mu_wake = transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt, free_wake=free_wake)
    
    else:
        mu, sigma, mu_wake = transient_solver(mesh_dict, wake_mesh_dict, num_nodes, nt, num_tot_panels, dt)
    
    

    






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