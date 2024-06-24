import numpy as np
import csdl_alpha as csdl 

def compute_source_strengths(mesh_dict, surface_names, num_nodes, nt, num_panels):
    sigma = csdl.Variable(shape=(num_nodes, nt, num_panels), value=0.)
    
    start, stop = 0, 0
    for surface in surface_names:
        num_surf_panels = mesh_dict[surface]['num_panels']
        stop += num_surf_panels

        coll_point_velocity = mesh_dict[surface]['coll_point_velocity']
        panel_normal = mesh_dict[surface]['panel_normal']

        vel_projection = csdl.einsum(coll_point_velocity, panel_normal, action='ijklm,ijklm->ijkl')

        sigma = sigma.set(csdl.slice[:,:,start:stop], value=csdl.reshape(vel_projection, shape=(num_nodes, nt, num_surf_panels)))

    return sigma # VECTORIZED in shape=(num_nodes, nt, num_surf_panels)

def compute_source_influence(dij, mij, dpij, dx, dy, dz, rk, ek, hk, sigma=1., mode='potential'):
    '''
    UPDATE TO EXPLAIN EACH INPUT
    each input is a list of length 4, holding the values at the corners
    ex: mij = [mij_i for i in range(4)] where each index i corresponds to a corner of the panel
    NOTE: dpij uses the same notation, but contains a sub-index to differentiate between x and y values
    ex: dpij[0][0] refers to the x-component of dpij for point 1
    '''
    if mode == 'potential':
        # source_influence = -sigma/4/np.pi*(
        #     ( # CHANGE TO USE dx, dy, dz
        #     ((dx[:,:,:,0]*dpij[:,:,:,0,1] - dy[:,:,:,0]*dpij[:,:,:,0,0])/(dij[:,:,:,0]+1.e-12)*csdl.log((rk[:,:,:,0] + rk[:,:,:,1] + dij[:,:,:,0]+1.e-12)/(rk[:,:,:,0] + rk[:,:,:,1] - dij[:,:,:,0] + 1.e-12))) + 
        #     ((dx[:,:,:,1]*dpij[:,:,:,1,1] - dy[:,:,:,1]*dpij[:,:,:,1,0])/(dij[:,:,:,1]+1.e-12)*csdl.log((rk[:,:,:,1] + rk[:,:,:,2] + dij[:,:,:,1]+1.e-12)/(rk[:,:,:,1] + rk[:,:,:,2] - dij[:,:,:,1] + 1.e-12))) + 
        #     ((dx[:,:,:,2]*dpij[:,:,:,2,1] - dy[:,:,:,2]*dpij[:,:,:,2,0])/(dij[:,:,:,2]+1.e-12)*csdl.log((rk[:,:,:,2] + rk[:,:,:,3] + dij[:,:,:,2]+1.e-12)/(rk[:,:,:,2] + rk[:,:,:,3] - dij[:,:,:,2] + 1.e-12))) + 
        #     ((dx[:,:,:,3]*dpij[:,:,:,3,1] - dy[:,:,:,3]*dpij[:,:,:,3,0])/(dij[:,:,:,3]+1.e-12)*csdl.log((rk[:,:,:,3] + rk[:,:,:,0] + dij[:,:,:,3]+1.e-12)/(rk[:,:,:,3] + rk[:,:,:,0] - dij[:,:,:,3] + 1.e-12)))
        # )
        # - dz[:,:,:,0] * (
        #     csdl.arctan((mij[:,:,:,0]*ek[:,:,:,0]-hk[:,:,:,0])/(dz[:,:,:,0]*rk[:,:,:,0]+1.e-12)) - csdl.arctan((mij[:,:,:,0]*ek[:,:,:,1]-hk[:,:,:,1])/(dz[:,:,:,0]*rk[:,:,:,1]+1.e-12)) + 
        #     csdl.arctan((mij[:,:,:,1]*ek[:,:,:,1]-hk[:,:,:,1])/(dz[:,:,:,1]*rk[:,:,:,1]+1.e-12)) - csdl.arctan((mij[:,:,:,1]*ek[:,:,:,2]-hk[:,:,:,2])/(dz[:,:,:,1]*rk[:,:,:,2]+1.e-12)) + 
        #     csdl.arctan((mij[:,:,:,2]*ek[:,:,:,2]-hk[:,:,:,2])/(dz[:,:,:,2]*rk[:,:,:,2]+1.e-12)) - csdl.arctan((mij[:,:,:,2]*ek[:,:,:,3]-hk[:,:,:,3])/(dz[:,:,:,2]*rk[:,:,:,3]+1.e-12)) + 
        #     csdl.arctan((mij[:,:,:,3]*ek[:,:,:,3]-hk[:,:,:,3])/(dz[:,:,:,3]*rk[:,:,:,3]+1.e-12)) - csdl.arctan((mij[:,:,:,3]*ek[:,:,:,0]-hk[:,:,:,0])/(dz[:,:,:,3]*rk[:,:,:,0]+1.e-12))
        # )) # note that dk[:,i,2] is the same for all i

        t1_y = dz[0]*dpij[0][0] * ((ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*rk[1] - (ek[1]*dpij[0][1]-hk[1]*dpij[0][0])*rk[0])
        t2_y = dz[1]*dpij[1][0] * ((ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*rk[2] - (ek[2]*dpij[1][1]-hk[2]*dpij[1][0])*rk[1])
        t3_y = dz[2]*dpij[2][0] * ((ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*rk[3] - (ek[3]*dpij[2][1]-hk[3]*dpij[2][0])*rk[2])
        t4_y = dz[3]*dpij[3][0] * ((ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*rk[0] - (ek[0]*dpij[3][1]-hk[0]*dpij[3][0])*rk[3])

        t1_x = dz[0]**2*rk[0]*rk[1]*dpij[0][0]**2 + (ek[0]*dpij[0][1]-hk[0]*dpij[0][0])*(ek[1]*dpij[0][1]-hk[1]*dpij[0][0])
        t2_x = dz[1]**2*rk[1]*rk[2]*dpij[1][0]**2 + (ek[1]*dpij[1][1]-hk[1]*dpij[1][0])*(ek[2]*dpij[1][1]-hk[2]*dpij[1][0])
        t3_x = dz[2]**2*rk[2]*rk[3]*dpij[2][0]**2 + (ek[2]*dpij[2][1]-hk[2]*dpij[2][0])*(ek[3]*dpij[2][1]-hk[3]*dpij[2][0])
        t4_x = dz[3]**2*rk[3]*rk[0]*dpij[3][0]**2 + (ek[3]*dpij[3][1]-hk[3]*dpij[3][0])*(ek[0]*dpij[3][1]-hk[0]*dpij[3][0])

        # atan2_1 = 2*csdl.arctan(t1_y / ((t1_x**2 + t1_y**2 + 1.e-12)**0.5 + t1_x))
        # atan2_2 = 2*csdl.arctan(t2_y / ((t2_x**2 + t2_y**2 + 1.e-12)**0.5 + t2_x))
        # atan2_3 = 2*csdl.arctan(t3_y / ((t3_x**2 + t3_y**2 + 1.e-12)**0.5 + t3_x))
        # atan2_4 = 2*csdl.arctan(t4_y / ((t4_x**2 + t4_y**2 + 1.e-12)**0.5 + t4_x))

        atan2_1 = 2*csdl.arctan(((t1_x**2 + t1_y**2)**0.5 - t1_x) / (t1_y + 1.e-12))
        atan2_2 = 2*csdl.arctan(((t2_x**2 + t2_y**2)**0.5 - t2_x) / (t2_y + 1.e-12))
        atan2_3 = 2*csdl.arctan(((t3_x**2 + t3_y**2)**0.5 - t3_x) / (t3_y + 1.e-12))
        atan2_4 = 2*csdl.arctan(((t4_x**2 + t4_y**2)**0.5 - t4_x) / (t4_y + 1.e-12))

        source_influence = -sigma/4/np.pi*(
            ( # CHANGE TO USE dx, dy, dz
            ((dx[0]*dpij[0][1] - dy[0]*dpij[0][0])/(dij[0]+1.e-12)*csdl.log((rk[0] + rk[1] + dij[0]+1.e-12)/(rk[0] + rk[1] - dij[0] + 1.e-12))) + 
            ((dx[1]*dpij[1][1] - dy[1]*dpij[1][0])/(dij[1]+1.e-12)*csdl.log((rk[1] + rk[2] + dij[1]+1.e-12)/(rk[1] + rk[2] - dij[1] + 1.e-12))) + 
            ((dx[2]*dpij[2][1] - dy[2]*dpij[2][0])/(dij[2]+1.e-12)*csdl.log((rk[2] + rk[3] + dij[2]+1.e-12)/(rk[2] + rk[3] - dij[2] + 1.e-12))) + 
            ((dx[3]*dpij[3][1] - dy[3]*dpij[3][0])/(dij[3]+1.e-12)*csdl.log((rk[3] + rk[0] + dij[3]+1.e-12)/(rk[3] + rk[0] - dij[3] + 1.e-12)))
        )
        - dz[0] * ( 
            # atan2_1 + atan2_2 + atan2_3 + atan2_4
            csdl.arctan((mij[0]*ek[0]-hk[0])/(dz[0]*rk[0]+1.e-12)) - csdl.arctan((mij[0]*ek[1]-hk[1])/(dz[0]*rk[1]+1.e-12)) + 
            csdl.arctan((mij[1]*ek[1]-hk[1])/(dz[1]*rk[1]+1.e-12)) - csdl.arctan((mij[1]*ek[2]-hk[2])/(dz[1]*rk[2]+1.e-12)) + 
            csdl.arctan((mij[2]*ek[2]-hk[2])/(dz[2]*rk[2]+1.e-12)) - csdl.arctan((mij[2]*ek[3]-hk[3])/(dz[2]*rk[3]+1.e-12)) + 
            csdl.arctan((mij[3]*ek[3]-hk[3])/(dz[3]*rk[3]+1.e-12)) - csdl.arctan((mij[3]*ek[0]-hk[0])/(dz[3]*rk[0]+1.e-12))
        )) # note that dk[:,i,2] is the same for all i
        return source_influence

    elif mode == 'velocity':
        # vel = csdl.Variable(shape=dz.shape[],value=0.)
        u = sigma/(4*np.pi) * (
            dpij[0][1]/dij[0]*csdl.log((rk[0] + rk[1] - dij[0])/(rk[0] + rk[1] + dij[0])) + 
            dpij[1][1]/dij[1]*csdl.log((rk[1] + rk[2] - dij[1])/(rk[1] + rk[2] + dij[1])) + 
            dpij[2][1]/dij[2]*csdl.log((rk[2] + rk[3] - dij[2])/(rk[2] + rk[3] + dij[2])) + 
            dpij[3][1]/dij[3]*csdl.log((rk[3] + rk[0] - dij[3])/(rk[3] + rk[0] + dij[3]))
        )

        v = sigma/(4*np.pi) * (
            dpij[0][0]/dij[0]*csdl.log((rk[0] + rk[1] - dij[0])/(rk[0] + rk[1] + dij[0])) + 
            dpij[1][0]/dij[1]*csdl.log((rk[1] + rk[2] - dij[1])/(rk[1] + rk[2] + dij[1])) + 
            dpij[2][0]/dij[2]*csdl.log((rk[2] + rk[3] - dij[2])/(rk[2] + rk[3] + dij[2])) + 
            dpij[3][0]/dij[3]*csdl.log((rk[3] + rk[0] - dij[3])/(rk[3] + rk[0] + dij[3]))
        ) * -1. # NOTE: THIS -1 IS BECAUSE OF THE DIFFERENT SIGN IN THE POTENTIAL EQUATION

        w = sigma/(4*np.pi) * (
            csdl.arctan((mij[0]*ek[0]-hk[0])/(dz[0]*rk[0]+1.e-12)) - csdl.arctan((mij[0]*ek[1]-hk[1])/(dz[0]*rk[1]+1.e-12)) + 
            csdl.arctan((mij[1]*ek[1]-hk[1])/(dz[1]*rk[1]+1.e-12)) - csdl.arctan((mij[1]*ek[2]-hk[2])/(dz[1]*rk[2]+1.e-12)) + 
            csdl.arctan((mij[2]*ek[2]-hk[2])/(dz[2]*rk[2]+1.e-12)) - csdl.arctan((mij[2]*ek[3]-hk[3])/(dz[2]*rk[3]+1.e-12)) + 
            csdl.arctan((mij[3]*ek[3]-hk[3])/(dz[3]*rk[3]+1.e-12)) - csdl.arctan((mij[3]*ek[0]-hk[0])/(dz[3]*rk[0]+1.e-12))
        )
        
        return u, v, w

def compute_source_AIC(mesh_dict, num_nodes, nt, num_tot_panels):
    surface_names = list(mesh_dict.keys())
    num_surfaces = len(surface_names)

    AIC = csdl.Variable(shape=(num_nodes, nt, num_tot_panels, num_tot_panels), value=0.)

    '''
    we want to capture interactions of the LOOP SURFACE j on MAIN SURFACE i
    we evaluate the induced potential at the evaluation point of surface i based on source elements from surface j
    for the main surface i, we need the center (collocation) point
    for the loop surface j, we need parameters of the panels (as these provide the influence)
    '''
    start_i, stop_i = 0, 0
    for i in range(num_surfaces):
        surf_i_name = surface_names[i]
        coll_point_i = mesh_dict[surf_i_name]['panel_center'] # evaluation point
        nc_i, ns_i = mesh_dict[surf_i_name]['nc'], mesh_dict[surf_i_name]['ns']
        num_panels_i = mesh_dict[surf_i_name]['num_panels']
        stop_i += num_panels_i

        start_j, stop_j = 0, 0
        for j in range(num_surfaces):
            surf_j_name = surface_names[j]
            nc_j, ns_j = mesh_dict[surf_j_name]['nc'], mesh_dict[surf_j_name]['ns']
            num_panels_j = mesh_dict[surf_j_name]['num_panels']
            stop_j += num_panels_j

            mesh_j = mesh_dict[surf_j_name]['mesh']
            panel_corners_j = mesh_dict[surf_j_name]['panel_corners']
            panel_x_dir_j = mesh_dict[surf_j_name]['panel_x_dir']
            panel_y_dir_j = mesh_dict[surf_j_name]['panel_y_dir']
            panel_normal_j = mesh_dict[surf_j_name]['panel_normal']
            dpij_j = mesh_dict[surf_j_name]['dpij']
            dij_j = mesh_dict[surf_j_name]['dij']
            mij_j = mesh_dict[surf_j_name]['mij']

            num_interactions = num_panels_i*num_panels_j

            coll_point_i_exp = csdl.expand(coll_point_i, (num_nodes, nt, nc_i-1, ns_i-1, num_panels_j, 4, 3), 'ijklm->ijklabm')
            coll_point_i_exp_vec = coll_point_i_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

            panel_corners_j_exp = csdl.expand(panel_corners_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklmn->ijaklmn')
            panel_corners_j_exp_vec = panel_corners_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))

            panel_x_dir_j_exp = csdl.expand(panel_x_dir_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
            panel_x_dir_j_exp_vec = panel_x_dir_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))
            panel_y_dir_j_exp = csdl.expand(panel_y_dir_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
            panel_y_dir_j_exp_vec = panel_y_dir_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))
            panel_normal_j_exp = csdl.expand(panel_normal_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 3), 'ijklm->ijaklbm')
            panel_normal_j_exp_vec = panel_normal_j_exp.reshape((num_nodes, nt, num_interactions, 4, 3))


            dpij_j_exp = csdl.expand(dpij_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4, 2), 'ijklmn->ijaklmn')
            dpij_j_exp_vec = dpij_j_exp.reshape((num_nodes, nt, num_interactions, 4, 2))
            dij_j_exp = csdl.expand(dij_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4), 'ijklm->ijaklm')
            dij_j_exp_vec = dij_j_exp.reshape((num_nodes, nt, num_interactions, 4))
            mij_j_exp = csdl.expand(mij_j, (num_nodes, nt, num_panels_i, nc_j-1, ns_j-1, 4), 'ijklm->ijaklm')
            mij_j_exp_vec = mij_j_exp.reshape((num_nodes, nt, num_interactions, 4))

            dp = coll_point_i_exp_vec - panel_corners_j_exp_vec
            # NOTE: consider changing dx,dy,dz and store in just dk with a dimension for x,y,z
            sum_ind = len(dp.shape) - 1
            dx = csdl.sum(dp*panel_x_dir_j_exp_vec, axes=(sum_ind,))
            dy = csdl.sum(dp*panel_y_dir_j_exp_vec, axes=(sum_ind,))
            dz = csdl.sum(dp*panel_normal_j_exp_vec, axes=(sum_ind,))
            rk = (dx**2 + dy**2 + dz**2)**0.5
            ek = dx**2 + dz**2
            hk = dx*dy

            source_influence_vec = compute_source_influence(dij_j_exp_vec, mij_j_exp_vec, dpij_j_exp_vec, dx, dy, dz, rk, ek, hk)
            source_influence = source_influence_vec.reshape((num_nodes, nt, num_panels_i, num_panels_j))

            AIC = AIC.set(csdl.slice[:,:,start_i:stop_i, start_j:stop_j], value=source_influence)
            start_j += num_panels_j
        start_i += num_panels_i

    return AIC