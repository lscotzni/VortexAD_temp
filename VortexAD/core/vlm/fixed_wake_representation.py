import numpy as np 
import csdl_alpha as csdl 

def fixed_wake_representation(mesh_dict, V_inf, num_panels=2):
    '''
    We are representing the wake here coming off of the trailing edge of the bound vortex grid.
    The wake will be "propagated" based on V_inf for the specified number of panels.
    '''
    surface_keys = mesh_dict.keys()
    dt = 100.
    for key in surface_keys:
        bd_vortex_grid_TE = mesh_dict[key]['bound_vortex_mesh'][:,-1,:,:]
        num_nodes = bd_vortex_grid_TE.shape[0]
        ns = bd_vortex_grid_TE.shape[1] # nc dimension gets removed so it's 1, not 2 for ns

        wake_vortex_mesh = csdl.Variable(shape=(num_nodes, num_panels, ns, 3), value=0.)
        for i in csdl.frange(num_panels):
            dx_i = csdl.expand(V_inf*dt*(i+1), (num_nodes, ns, 3), 'i->abi')
            wake_vortex_mesh = wake_vortex_mesh.set(csdl.slice[:,i,:,:], value=dx_i) 

        mesh_dict[key]['wake_vortex_mesh'] = wake_vortex_mesh # we do NOT add the bound vortex TE here so we don't duplicate information

    return mesh_dict