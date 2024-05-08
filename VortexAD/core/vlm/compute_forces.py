import numpy as np
import csdl_alpha as csdl 

def compute_forces(num_nodes, mesh_dict, output_dict, V_inf, alpha, ref_point='default'):
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)
    for i in range(num_surfaces):
        surface_name = surface_names[i]
        nc, ns = mesh_dict[surface_name]['nc'], mesh_dict[surface_name]['ns']
        net_gamma = output_dict[surface_name]['net_gamma'] # (num_nodes, nc-1, ns-1)
        v_induced = output_dict[surface_name]['force_pts_v_induced']
        panel_area = mesh_dict[surface_name]['panel_area']
        bound_vec = mesh_dict[surface_name]['bound_vec']

        target_shape = (num_nodes, nc-1, ns-1, 3)

        V_inf_exp = csdl.expand(V_inf, target_shape, 'ij->iabj')
        net_gamma_exp = csdl.expand(net_gamma, target_shape, 'ijk->ijka')
        panel_area_exp = csdl.expand(panel_area, target_shape, 'ijk->ijka')
        v_total = V_inf_exp + v_induced
        # v_total = V_inf_exp
        rho=1.225

        panel_forces = net_gamma_exp*csdl.cross(v_total, bound_vec, axis=3) * rho
        output_dict[surface_name]['total_forces'] = panel_forces

        panel_forces_x = panel_forces[:,:,:,0]
        panel_forces_y = panel_forces[:,:,:,1]
        panel_forces_z = panel_forces[:,:,:,2]

        surface_area = csdl.sum(panel_area, axes=(1,2))
        cosa = csdl.expand(csdl.cos(alpha), panel_forces_x.shape, 'i->iab')
        sina = csdl.expand(csdl.sin(alpha), panel_forces_x.shape, 'i->iab')
        # cosa = csdl.expand(csdl.cos(alpha), panel_forces_x.shape)
        # sina = csdl.expand(csdl.sin(alpha), panel_forces_x.shape)

        panel_lift = panel_forces_z*cosa - panel_forces_x*sina
        panel_drag = panel_forces_z*sina + panel_forces_x*cosa

        spanwise_sec_lift = csdl.sum(panel_lift, axes=(1,)) # summing across chordwise direction
        spanwise_sec_drag = csdl.sum(panel_drag, axes=(1,)) # summing across chordwise direction

        surface_lift = csdl.sum(spanwise_sec_lift, axes=(1,)) # summing across spanwise direction 
        surface_drag = csdl.sum(spanwise_sec_drag, axes=(1,)) # summing across spanwise direction 

        CL = surface_lift/(0.5*rho*csdl.norm(V_inf, axes=(1,))**2*surface_area)
        CDi = surface_drag/(0.5*rho*csdl.norm(V_inf, axes=(1,))**2*surface_area)

        output_dict[surface_name]['L'] = surface_lift
        output_dict[surface_name]['Di'] = surface_drag
        output_dict[surface_name]['CL'] = CL
        output_dict[surface_name]['CDi'] = CDi

        # total force and moment w.r.t. input reference point in the body-fixed frame
        total_force = csdl.sum(panel_forces, axes=(1,2))
        if ref_point == 'default':

            ref_point_expanded = csdl.expand(
                csdl.Variable(value=np.array([0.,0.,0.])),
                panel_forces.shape, 
                'i->abci'
            )
        else:
            ref_point_expanded = csdl.expand(ref_point, panel_forces.shape, 'i->abci')

        panel_moment = csdl.cross(ref_point_expanded, panel_forces, axis=3)
        total_moment = csdl.sum(panel_moment, axes=(1,2))

        output_dict[surface_name]['total_force'] = total_force
        output_dict[surface_name]['total_moment'] = total_moment
        

    return output_dict



'''
Use forces to:
- compute lift and drag for each surface
- compute CL and CDi for each surface

- compute total lift and drag for entire system
'''