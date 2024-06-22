import numpy as np
import csdl_alpha as csdl 

def compute_forces(num_nodes, mesh_dict, output_dict, V_inf=None, alpha_ML=None, 
               airfoil_Cl_models=None,
               airfoil_Cd_models=None,
               airfoil_Cp_models=None,
               airfoil_alpha_stall_models=None,
               reynolds_numbers=None,
               chord_length_mid_panel=None,
               ref_point='default'):
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    surface_force = []
    surface_moment = []
    surface_lift = []
    surface_drag = []

    surface_panel_forces = []
    surface_panel_force_points = []
    surface_sectional_cop = []
    surface_cop = [] 
    surface_CL = []
    surface_CDi = []
    surface_spanwise_Cp = []

    total_force = csdl.Variable(shape=(num_nodes, 3), value=0.)
    total_moment = csdl.Variable(shape=(num_nodes, 3), value=0.)
    total_lift = csdl.Variable(shape=(num_nodes, ), value=0.)
    total_drag = csdl.Variable(shape=(num_nodes, ), value=0.)

    for i in range(num_surfaces):
        surface_name = surface_names[i]
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        panel_area = mesh_dict[surface_name]['panel_area']
        bound_vec = mesh_dict[surface_name]['bound_vec']
        bound_vec_velocity = mesh_dict[surface_name]['bound_vector_velocity']
        force_eval_pts = mesh_dict[surface_name]['force_eval_points']
        normal_vec = mesh_dict[surface_name]['bd_normal_vec']

        V_inf = csdl.average(bound_vec_velocity, axes=(1, )) #csdl.sum(bound_vec_velocity[:,:,:,:], axes=(1,)) / (ns-1) # AXIS 1 HERE BECAUSE THE SHAPE LENGTH DECREASES BY 1 B/C THE ORIGINAL AXIS 1 DISAPPEARS
        alpha_surf = csdl.arctan(V_inf[:,:,2]/V_inf[:,:, 0])

        # if num_nodes == 1:
        #     alpha_surf_exp = csdl.expand(alpha_surf, normal_vec.shape[:-1])
        # else:
        alpha_surf_exp = csdl.expand(alpha_surf, normal_vec.shape[:-1], 'ik->ijk')

        # NOTE: ADJUST alpha_ML INPUT TO BE A LIST CORRESPONDING TO THE NUMBER OF SURFACES
        # NOTE: ADJUST V_inf ABOVE BY ROTATING IT BY alpha_ML AT THE BEGINNING OF THE SOLVER (needed for no-penetration condition)
        if alpha_ML[i] is not None: # alpha_ML would come in with shape (num_nodes, ns-1); need to expand to (num_nodes, nc, ns)
            alpha_ML_exp = csdl.expand(alpha_ML[i], alpha_surf_exp.shape, 'ij->iaj')
            alpha_tot = alpha_surf_exp + alpha_ML_exp
        else:
            alpha_tot = alpha_surf_exp
        net_gamma = output_dict[surface_name]['net_gamma'] # (num_nodes, nc-1, ns-1)
        v_induced = output_dict[surface_name]['force_pts_v_induced']

        target_shape = (num_nodes, nc-1, ns-1, 3)

        V_inf_exp = csdl.expand(V_inf, target_shape, 'ijk->iajk')
        net_gamma_exp = csdl.expand(net_gamma, target_shape, 'ijk->ijka')
        panel_area_exp = csdl.expand(panel_area, target_shape, 'ijk->ijka')
        # v_total = V_inf_exp + v_induced
        v_total = bound_vec_velocity + v_induced
        # print('v_induced:')
        # print(v_induced.value)
        # print(v_total.value)
        # exit()

        rho=1.225

        panel_forces = net_gamma_exp*csdl.cross(v_total, bound_vec, axis=3) * rho
        output_dict[surface_name]['total_forces'] = panel_forces
        # print('panel_forces:')
        # print(panel_forces.value)
        # exit()
        if alpha_ML[i] is not None:
            alpha = alpha_ML_exp
            # print(alpha.shape)
            rotation_tensor = csdl.Variable(shape=alpha.shape + (3, 3), value=0.)
            rotation_tensor = rotation_tensor.set(csdl.slice[:, :, :, 0, 0], csdl.cos(alpha))
            rotation_tensor = rotation_tensor.set(csdl.slice[:, :, :, 0, 2], -csdl.sin(alpha))
            rotation_tensor = rotation_tensor.set(csdl.slice[:, :, :, 1, 1], 1)
            rotation_tensor = rotation_tensor.set(csdl.slice[:, :, :, 2, 0], csdl.sin(alpha))
            rotation_tensor = rotation_tensor.set(csdl.slice[:, :, :, 2, 2], csdl.cos(alpha))

            panel_forces = csdl.einsum(rotation_tensor, panel_forces, action='ijkml,ijkl->ijkm')

            panel_forces_x = panel_forces[:,:,:,0]
            panel_forces_y = panel_forces[:,:,:,1]
            panel_forces_z = panel_forces[:,:,:,2]
            
            # exit()
        else:
            panel_forces_x = panel_forces[:,:,:,0]
            panel_forces_y = panel_forces[:,:,:,1]
            panel_forces_z = panel_forces[:,:,:,2]

        surface_area = csdl.sum(panel_area, axes=(1,2))
        spanwise_areas = csdl.sum(panel_area, axes=(1, ))
        # print(V_inf.value)
        # exit()
        # print(np.rad2deg(alpha_surf_exp.value))
        # print(np.rad2deg(alpha_ML_exp.value))
        # print(np.rad2deg(alpha_tot.value))
        cosa = csdl.cos(alpha_tot)
        sina = csdl.sin(alpha_tot)

        panel_lift = panel_forces_z*cosa - panel_forces_x*sina
        panel_drag = panel_forces_z*sina + panel_forces_x*cosa

        spanwise_sec_lift = csdl.sum(panel_lift, axes=(1,)) # summing across chordwise direction
        spanwise_Cl = spanwise_sec_lift / (0.5 * rho * csdl.norm(V_inf, axes=(2, ))**2 * spanwise_areas) * -1
        Ma = csdl.Variable(shape=spanwise_sec_lift.shape, value=0.18)

        if airfoil_alpha_stall_models[i] is not None:
            alpha_min_max = airfoil_alpha_stall_models[i].evaluate(alpha=Ma, Re=reynolds_numbers[i], Ma=Ma)
            alpha_implicit = csdl.ImplicitVariable(shape=spanwise_sec_lift.shape, value=0)
            Cl = airfoil_Cl_models[i].evaluate(alpha=alpha_implicit, Re=reynolds_numbers[i], Ma=Ma)
            lower_bracket = alpha_min_max[i, :, 0].reshape(Cl.shape)
            upper_bracket = alpha_min_max[i, :, 1].reshape(Cl.shape)
            residual = Cl - spanwise_Cl
            solver = csdl.nonlinear_solvers.bracketed_search.BracketedSearch(elementwise_states=True)
            solver.add_state(alpha_implicit, residual, bracket=(lower_bracket, upper_bracket))
            solver.run()

            print("ALPHA", np.rad2deg(alpha_implicit.value))


        if airfoil_Cd_models[i] is not None:
            chord_lengths = chord_length_mid_panel[i]
            Cd = airfoil_Cd_models[i].evaluate(alpha=alpha_implicit, Re=reynolds_numbers[i], Ma=Ma)
            viscous_drag = Cd *  0.5 * rho * csdl.norm(V_inf, axes=(2, ))**2 * chord_lengths

        if airfoil_Cp_models[i] is not None:
            Cp = airfoil_Cp_models[i].evaluate(alpha=alpha_implicit, Re=reynolds_numbers[i], Ma=Ma)
            surface_spanwise_Cp.append(Cp)
        else:
            surface_spanwise_Cp.append(None)

        spanwise_sec_drag = csdl.sum(panel_drag, axes=(1,)) # summing across chordwise direction

        lift_surf = csdl.sum(spanwise_sec_lift, axes=(1,)) # summing across spanwise direction 
        drag_surf = csdl.sum(spanwise_sec_drag, axes=(1,)) # summing across spanwise direction 

        CL = lift_surf/(0.5*rho*csdl.average(csdl.norm(V_inf, axes=(2,)), axes=(1, ))**2*surface_area)
        CDi = drag_surf/(0.5*rho*csdl.average(csdl.norm(V_inf, axes=(2,)), axes=(1, ))**2*surface_area)

        output_dict[surface_name]['L'] = lift_surf
        output_dict[surface_name]['Di'] = drag_surf
        output_dict[surface_name]['CL'] = CL
        output_dict[surface_name]['CDi'] = CDi

        # total force and moment w.r.t. input reference point in the body-fixed frame
        total_force_surf = csdl.sum(panel_forces, axes=(1,2))
        if ref_point == 'default':

            ref_point_expanded = csdl.expand(
                csdl.Variable(value=np.array([0.,0.,0.])),
                panel_forces.shape, 
                'i->abci'
            )
        else:
            ref_point_expanded = csdl.expand(ref_point, panel_forces.shape, 'i->abci')
        moment_arm_surf = force_eval_pts - ref_point_expanded
        panel_moment_surf = csdl.cross(moment_arm_surf, panel_forces, axis=3)
        total_moment_surf = csdl.sum(panel_moment_surf, axes=(1,2))

        output_dict[surface_name]['force'] = total_force_surf
        output_dict[surface_name]['moment'] = total_moment_surf

        # center of pressure calculation
        # compute normal forces
        # numerator: normal force * force_eval_pts (expanding the normal force)
        # denominator: sum of all normal forces
        normal_force = csdl.sum(panel_forces*normal_vec, axes=(3,))
        normal_force_exp = csdl.expand(normal_force, normal_vec.shape, 'ijk->ijkl')
        normal_force_sum_exp = csdl.sum(csdl.expand(normal_force, normal_force.shape + (3,), 'ijk->ijka'), axes=(1,2)) # shape (num_nodes, 3)
    
        cop = csdl.sum(moment_arm_surf*normal_force_exp, axes=(1,2))/normal_force_sum_exp

        sectional_cop = csdl.sum(moment_arm_surf*normal_force_exp, axes=(1,)) / csdl.sum(csdl.expand(normal_force, normal_force.shape + (3,), 'ijk->ijka'), axes=(1,))

        surface_panel_forces.append(panel_forces)
        surface_panel_force_points.append(force_eval_pts)
        surface_force.append(total_force_surf)
        surface_moment.append(total_moment_surf)
        surface_lift.append(lift_surf)
        surface_drag.append(drag_surf)

        surface_CL.append(CL)
        surface_CDi.append(CDi)
        surface_cop.append(cop)
        surface_sectional_cop.append(sectional_cop)

        total_force = total_force + total_force_surf
        total_moment = total_moment + total_moment_surf
        total_lift = total_lift + lift_surf
        total_drag = total_drag + drag_surf

    surface_output_dict = {
        'surface_panel_forces': surface_panel_forces,
        'surface_panel_force_points': surface_panel_force_points,
        'surface_force': surface_force,
        'surface_moment': surface_moment,
        'surface_lift': surface_lift,
        'surface_drag': surface_drag,
        'surface_CL': surface_CL,
        'surface_CDi': surface_CDi, 
        'surface_cop': surface_cop,
        'surface_sectional_cop': surface_sectional_cop,
        'surface_spanwise_Cp' : surface_spanwise_Cp,
    }

    total_output_dict = {
        'total_force': total_force,
        'total_moment': total_moment,
        'total_lift': total_lift,
        'total_drag': total_drag,
    }
    
    return surface_output_dict, total_output_dict



'''
Use forces to:
- compute lift and drag for each surface
- compute CL and CDi for each surface

- compute total lift and drag for entire system
'''