import csdl_alpha as csdl 

from VortexAD.core.vlm.velocity_computations import compute_normal_velocity, compute_induced_velocity

def setup_linear_system(num_nodes, mesh_dict, V_inf):
    '''
    Assemble the AIC matrix and RHS boundary condition here
    '''
    surface_names = list(mesh_dict.keys())
    print(surface_names)
    num_surfaces = len(surface_names)
    print(num_surfaces)

    num_total_panels = 0
    for key in mesh_dict.keys():
        ns, nc = mesh_dict[key]['ns'], mesh_dict[key]['nc']
        num_total_panels += (ns-1)*(nc-1)

    RHS = csdl.Variable(shape=(num_nodes, num_total_panels), name='RHS', value=0.)
    AIC = csdl.Variable(shape=(num_nodes, num_total_panels, num_total_panels,), name='AIC', value=0.)

    '''
    What's needed here:
    - bound vortex grid/mesh
    - collocation points

    What we want to find:
    - induced velocity from all panels on panel i
    '''
    start_panel_counter = 0
    stop_panel_counter = 0
    for i in range(num_surfaces):
        surface_name = surface_names[i] # THIS IS THE SURFACE OF INTEREST
        ns_i, nc_i = mesh_dict[surface_name]['ns'], mesh_dict[surface_name]['nc']
        np_surf_i = (ns_i-1)*(nc_i-1)
        bd_vortex_normals_i = mesh_dict[surface_name]['bd_normal_vec'] # (num_nodes, ns-1, nc-1, 3)
        collocation_points_i = mesh_dict[surface_name]['collocation_points'] # (num_nodes, ns-1, nc-1, 3)
        bound_vortex_mesh = mesh_dict[surface_name]['bound_vortex_mesh'] # (num_nodes, ns-1, nc-1, 3)

        # p1_bd = bound_vortex_mesh[:, :-1, :-1, :]
        # p2_bd = bound_vortex_mesh[:, :-1, 1:, :]
        # p3_bd = bound_vortex_mesh[:, 1:, 1:, :]
        # p4_bd = bound_vortex_mesh[:, 1:, :-1, :]

        # bd_vortex_normals_vec = csdl.reshape(bd_vortex_normals, shape=(num_nodes, np_surf, 3))
        collocation_points_vec = csdl.reshape(collocation_points_i, shape=(num_nodes, np_surf_i, 3))

        # p1_bd_vec = csdl.reshape(p1_bd, (num_nodes, np_surf, 3))
        # p2_bd_vec = csdl.reshape(p2_bd, (num_nodes, np_surf, 3))
        # p3_bd_vec = csdl.reshape(p3_bd, (num_nodes, np_surf, 3))
        # p4_bd_vec = csdl.reshape(p4_bd, (num_nodes, np_surf, 3))

        # SETTING BOUNDARY CONDITION
        stop_panel_counter += np_surf_i
        surface_bc = compute_normal_velocity(V_inf, bd_vortex_normals_i)
        RHS = RHS.set(csdl.slice[:, start_panel_counter:stop_panel_counter], csdl.reshape(surface_bc, (num_nodes, np_surf_i)))

        # getting interactions of looping surface j ON main surface i
        # means we evaluate induced velocity at collocation point of surface i
        # based on the influence from the bound vortices from surface j
        start_panel_counter_j = 0
        stop_panel_counter_j = 0
        for j in range(num_surfaces):
            surface_name_j = surface_names[j]
            ns_j, nc_j = mesh_dict[surface_name_j]['ns'], mesh_dict[surface_name_j]['nc']
            np_surf_j = (ns_j-1)*(nc_j-1)

            bound_vortex_mesh_j = mesh_dict[surface_name_j]['bound_vortex_mesh']
            bd_vortex_normals_j = mesh_dict[surface_name_j]['bd_normal_vec']

            # VECTORIZE AND EXPAND EVERYTHING TO SHAPE (num_nodes, np_surf_j*np_surf_i) (+ (3,) if needed)
            num_interactions = np_surf_i*np_surf_j

            p1_bd = csdl.expand(bound_vortex_mesh_j[:, :-1, :-1, :], (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p2_bd = csdl.expand(bound_vortex_mesh_j[:, :-1, 1:, :], (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p3_bd = csdl.expand(bound_vortex_mesh_j[:, 1:, 1:, :], (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')
            p4_bd = csdl.expand(bound_vortex_mesh_j[:, 1:, :-1, :], (num_nodes, nc_j-1, ns_j-1, np_surf_i, 3), 'ijkl->ijkal')

            interaction_shape = (num_nodes, num_interactions, 3)

            p1_bd_vec = p1_bd.reshape(interaction_shape)
            p2_bd_vec = p2_bd.reshape(interaction_shape)
            p3_bd_vec = p3_bd.reshape(interaction_shape)
            p4_bd_vec = p4_bd.reshape(interaction_shape)

            # use csdl for loop to iteratively expand over one variable (different expansion from csdl.expand)
            coll_point_i_exp = csdl.Variable(shape=(num_nodes, np_surf_j, nc_i-1, ns_i-1, 3), value=0.)
            bd_normal_vec_i_exp = csdl.Variable(shape=(num_nodes, np_surf_j, nc_i-1, ns_i-1, 3), value=0.)
            for k in csdl.frange((np_surf_j)):
                coll_point_i_exp = coll_point_i_exp.set(csdl.slice[:,k,:,:,:], value=collocation_points_i)
                bd_normal_vec_i_exp = bd_normal_vec_i_exp.set(csdl.slice[:,k,:,:,:], value=bd_vortex_normals_i)

            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, num_interactions, 3))
            
            v_i_12 = compute_induced_velocity(p1_bd_vec, p2_bd_vec, coll_point_i_exp_vec)
            v_i_23 = compute_induced_velocity(p2_bd_vec, p3_bd_vec, coll_point_i_exp_vec)
            v_i_34 = compute_induced_velocity(p3_bd_vec, p4_bd_vec, coll_point_i_exp_vec)
            v_i_41 = compute_induced_velocity(p4_bd_vec, p1_bd_vec, coll_point_i_exp_vec)

            v_induced = v_i_12 + v_i_23 + v_i_34 + v_i_41

            v_induced_grid = v_induced.reshape((num_nodes, np_surf_j, np_surf_i, 3))

            bd_normal_vec_i_exp_grid = bd_normal_vec_i_exp.reshape((num_nodes, np_surf_j, np_surf_i, 3))

            normal_induced_vel = csdl.sum(v_induced_grid*bd_normal_vec_i_exp_grid, axes=(3,))

            # normal_induced_vel = csdl.sum(v_induced_grid, bd_vortex_normals_i, ([3], [3]))

            stop_panel_counter_j += np_surf_j
            AIC = AIC.set(csdl.slice[:, start_panel_counter_j:stop_panel_counter_j, start_panel_counter:stop_panel_counter], value=normal_induced_vel)
            # AIC[:, start_panel_counter_j:stop_panel_counter_j, start_panel_counter:stop_panel_counter] = normal_induced_vel
            start_panel_counter_j += np_surf_j
        start_panel_counter += np_surf_i

    
    
    '''
    NOTE: csdl for loops work differently from python ones
    - the indices during slicing MUST be the same within the for loop
    - looping iterator is not an integer but a CSDL variable
        - we can't use .get(i) or var[i] unless the operation is done on a CSDL variable
        - ex: surface_names[i] would fail
    
    To capture the interactions across surfaces, we need two loops:
    - outer python for-loop to assemble indices, etc.
        - inner csdl for-loop to assign values to compute interactions
    '''

    return AIC, RHS