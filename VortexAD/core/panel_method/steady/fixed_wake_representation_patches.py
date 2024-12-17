import numpy as np
import csdl_alpha as csdl

def fixed_wake_representation_patches(mesh_dict, num_nodes, wake_propagation_dt=100.):
    # wake propagation dt: time elapsed to propagate wake back (dx = V_inf*dt)
    surface_names = list(mesh_dict.keys())
    wake_mesh_dict = {}

    for i, surface_name in enumerate(surface_names):
        patch_names = list(mesh_dict[surface_name].keys())
        patch_type = [mesh_dict[surface_name][patch_name]['patch'] for patch_name in patch_names]

        if len(patch_type) == 2 and patch_type[0] is not None:
            surface_patch_mode = 'separated'
        elif len(patch_type) == 1 and patch_type[0] == 'wrap':
            surface_patch_mode = 'wrap'
        else:
            continue # patches are not wake-shedding
        
        surf_wake_mesh_dict = {}
        nc_w = 2 # only one panel for now
        if surface_patch_mode == 'wrap':
            surface_mesh = mesh_dict[surface_name][patch_names[0]]['mesh'] # (nn, nc, ns, 3)
            mesh_velocity = mesh_dict[surface_name][patch_names[0]]['nodal_velocity']

            ns = surface_mesh.shape[2]
            
            TE = (surface_mesh[:,0,:,:] + surface_mesh[:,-1,:,:])/2.
            wake_end = TE + mesh_velocity[:,-1,:,:]*wake_propagation_dt

        elif surface_patch_mode == 'separated':
            surf_mesh_0 = mesh_dict[surface_name][patch_names[0]]['mesh']
            surf_mesh_vel_0 = mesh_dict[surface_name][patch_names[0]]['nodal_velocity']

            surf_mesh_1 = mesh_dict[surface_name][patch_names[1]]['mesh']
            surf_mesh_vel_1 = mesh_dict[surface_name][patch_names[1]]['nodal_velocity']
            # NOTE: at this point, we don't know which is upper or lower

            ns = surf_mesh_0.shape[2]

            TE = (surf_mesh_0[:,-1,:,:] + surf_mesh_1[:,-1,::-1,:]) / 2.
            TE_vel_avg = (surf_mesh_vel_0[:,-1,:,:] + surf_mesh_vel_1[:,-1,::-1,:]) / 2.
            wake_end = TE + TE_vel_avg*wake_propagation_dt

        wake_mesh = csdl.Variable(value=np.zeros((num_nodes, 2, ns, 3))) # only 2 nodes in the "chordwise" direction
        wake_mesh = wake_mesh.set(csdl.slice[:,0,:,:], value=TE)
        wake_mesh = wake_mesh.set(csdl.slice[:,1,:,:], value=wake_end)


        surf_wake_mesh_dict['mesh'] = wake_mesh
        surf_wake_mesh_dict['nc'], surf_wake_mesh_dict['ns'] = nc_w, ns
        surf_wake_mesh_dict['num_panels'] = (nc_w-1)*(ns-1)
        surf_wake_mesh_dict['num_points'] = nc_w*ns


        # computing wake parameters

        # mesh has shape (nn, 2, ns, 3) bc there's only ONE chordwise panel
        R1 = wake_mesh[:,:-1,:-1,:]
        R2 = wake_mesh[:,1:,:-1,:]
        R3 = wake_mesh[:,1:,1:,:]
        R4 = wake_mesh[:,:-1,1:,:]

        Rc = (R1+R2+R3+R4)/4.
        panel_center = Rc
        surf_wake_mesh_dict['panel_center'] = panel_center

        panel_corners = csdl.Variable(value=np.zeros((num_nodes, nc_w-1, ns-1, 4, 3)))
        panel_corners = panel_corners.set(csdl.slice[:,:,:,0,:], value=R1)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,1,:], value=R2)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,2,:], value=R3)
        panel_corners = panel_corners.set(csdl.slice[:,:,:,3,:], value=R4)
        surf_wake_mesh_dict['panel_corners'] = panel_corners

        D1 = R3-R1
        D2 = R4-R2

        D1D2_cross = csdl.cross(D1, D2, axis=3)
        D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(3,))
        panel_area = D1D2_cross_norm/2.
        surf_wake_mesh_dict['panel_area'] = panel_area

        normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'ijk->ijka')

        m_dir = (R3+R4)/2. - Rc
        m_norm = csdl.norm(m_dir, axes=(3,))
        m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'ijk->ijka')
        l_vec = csdl.cross(m_vec, normal_vec, axis=3)

        panel_x_dir = l_vec
        panel_y_dir = m_vec
        panel_normal = normal_vec

        surf_wake_mesh_dict['panel_x_dir'] = panel_x_dir
        surf_wake_mesh_dict['panel_y_dir'] = panel_y_dir
        surf_wake_mesh_dict['panel_normal'] = panel_normal

        # s = csdl.Variable(shape=(panel_corners.shape[0],) + panel_corners.shape[2:], value=0.)
        s = csdl.Variable(shape=panel_corners.shape, value=0.)
        s = s.set(csdl.slice[:,:,:,:-1,:], value=panel_corners[:,:,:,1:,:] - panel_corners[:,:,:,:-1,:])
        s = s.set(csdl.slice[:,:,:,-1,:], value=panel_corners[:,:,:,0,:] - panel_corners[:,:,:,-1,:])

        l_exp = csdl.expand(l_vec, s.shape, 'ijkl->ijkal')
        m_exp = csdl.expand(m_vec, s.shape, 'ijkl->ijkal')
        
        S = csdl.norm(s+1.e-12, axes=(4,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
        SL = csdl.sum(s*l_exp+1.e-12, axes=(4,))
        SM = csdl.sum(s*m_exp+1.e-12, axes=(4,))

        surf_wake_mesh_dict['S'] = S
        surf_wake_mesh_dict['SL'] = SL
        surf_wake_mesh_dict['SM'] = SM
        
        wake_mesh_dict[surface_name] = surf_wake_mesh_dict

    return wake_mesh_dict
