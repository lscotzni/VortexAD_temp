import numpy as np
import csdl_alpha as csdl

def pre_processor_new(mesh_dict, mode='structured'):
    surface_names = list(mesh_dict.keys())
    if mode == 'structured':
        for i, key in enumerate(surface_names):
            mesh = mesh_dict[key]['mesh']
            mesh_shape = mesh.shape
            nc, ns = mesh_shape[-3], mesh_shape[-2]

            mesh_dict[key]['num_panels'] = (nc-1)*(ns-1)
            mesh_dict[key]['nc'] = nc
            mesh_dict[key]['ns'] = ns

            R1 = mesh[:,:,:-1,:-1,:]
            R2 = mesh[:,:,1:,:-1,:]
            R3 = mesh[:,:,1:,1:,:]
            R4 = mesh[:,:,:-1,1:,:]
        
            S1 = (R1+R2)/2.
            S2 = (R2+R3)/2.
            S3 = (R3+R4)/2.
            S4 = (R4+R1)/2.

            Rc = (R1+R2+R3+R4)/4.
            mesh_dict[key]['panel_center'] = Rc

            panel_corners = csdl.Variable(shape=(Rc.shape[:-1] + (4,3)), value=0.)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,0,:], value=R1)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,1,:], value=R2)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,2,:], value=R3)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,3,:], value=R4)
            mesh_dict[key]['panel_corners'] = panel_corners

            D1 = R3-R1
            D2 = R4-R2

            D1D2_cross = csdl.cross(D1, D2, axis=4)
            D1D2_cross_norm = csdl.norm(D1D2_cross, axes=(4,))
            panel_area = D1D2_cross_norm/2.
            mesh_dict[key]['panel_area'] = panel_area

            normal_vec = D1D2_cross / csdl.expand(D1D2_cross_norm, D1D2_cross.shape, 'ijkl->ijkla')
            mesh_dict[key]['panel_normal'] = normal_vec

            panel_center_mod = Rc - normal_vec*0.001
            panel_center_mod = Rc
            mesh_dict[key]['panel_center_mod'] = panel_center_mod

            m_dir = S3 - Rc
            m_norm = csdl.norm(m_dir, axes=(4,))
            m_vec = m_dir / csdl.expand(m_norm, m_dir.shape, 'ijkl->ijkla')
            l_vec = csdl.cross(m_vec, normal_vec, axis=4)
            # this also tells us that normal_vec = cross(l_vec, m_vec)

            mesh_dict[key]['panel_x_dir'] = l_vec
            mesh_dict[key]['panel_y_dir'] = m_vec

            SMP = csdl.norm((S2)/2 - Rc, axes=(4,))
            SMQ = csdl.norm((S3)/2 - Rc, axes=(4,)) # same as m_norm

            mesh_dict[key]['SMP'] = SMP
            mesh_dict[key]['SMQ'] = SMQ

            s = csdl.Variable(shape=panel_corners.shape, value=0.)
            s = s.set(csdl.slice[:,:,:,:,:-1,:], value=panel_corners[:,:,:,:,1:,:] - panel_corners[:,:,:,:,:-1,:])
            s = s.set(csdl.slice[:,:,:,:,-1,:], value=panel_corners[:,:,:,:,0,:] - panel_corners[:,:,:,:,-1,:])

            l_exp = csdl.expand(l_vec, panel_corners.shape, 'ijklm->ijklam')
            m_exp = csdl.expand(m_vec, panel_corners.shape, 'ijklm->ijklam')
            
            S = csdl.norm(s+1.e-12, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
            # S = csdl.norm(s, axes=(5,)) # NOTE: ADD NUMERICAL SOFTENING HERE BECAUSE OVERLAPPING NODES WILL CAUSE THIS TO BE 0
            SL = csdl.sum(s*l_exp, axes=(5,))
            SM = csdl.sum(s*m_exp, axes=(5,))

            mesh_dict[key]['S'] = S
            mesh_dict[key]['SL'] = SL
            mesh_dict[key]['SM'] = SM

            delta_coll_point = csdl.Variable(Rc.shape[:-1] + (4,2), value=0.)
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,:,0,0], value=csdl.sum((Rc[:,:,:-1,:,:]-Rc[:,:,1:,:,:])*l_vec[:,:,1:,:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,:,0,1], value=csdl.sum((Rc[:,:,:-1,:,:]-Rc[:,:,1:,:,:])*m_vec[:,:,1:,:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,:,1,0], value=csdl.sum((Rc[:,:,1:,:,:]-Rc[:,:,:-1,:,:])*l_vec[:,:,:-1,:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,:,1,1], value=csdl.sum((Rc[:,:,1:,:,:]-Rc[:,:,:-1,:,:])*m_vec[:,:,:-1,:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,1:,2,0], value=csdl.sum((Rc[:,:,:,:-1,:]-Rc[:,:,:,1:,:])*l_vec[:,:,:,1:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,1:,2,1], value=csdl.sum((Rc[:,:,:,:-1,:]-Rc[:,:,:,1:,:])*m_vec[:,:,:,1:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,:-1,3,0], value=csdl.sum((Rc[:,:,:,1:,:]-Rc[:,:,:,:-1,:])*l_vec[:,:,:,:-1,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,:-1,3,1], value=csdl.sum((Rc[:,:,:,1:,:]-Rc[:,:,:,:-1,:])*m_vec[:,:,:,:-1,:], axes=(4,)))

            mesh_dict[key]['delta_coll_point'] = delta_coll_point

            nodal_vel = mesh_dict[key]['nodal_velocity']
            mesh_dict[key]['nodal_cp_velocity'] = (
                nodal_vel[:,:,:-1,:-1,:]+nodal_vel[:,:,:-1,1:,:]+\
                nodal_vel[:,:,1:,1:,:]+nodal_vel[:,:,1:,:-1,:]) / 4.

            # computing planform area
            panel_width_spanwise = csdl.norm((mesh[:,:,:,1:,:] - mesh[:,:,:,:-1,:]), axes=(4,))
            avg_panel_width_spanwise = csdl.average(panel_width_spanwise, axes=(2,)) # num_nodes, nt, ns - 1
            surface_TE = (mesh[:,:,-1,:-1,:] + mesh[:,:,0,:-1,:] + mesh[:,:,-1,1:,:] + mesh[:,:,0,1:,:])/4
            surface_LE = (mesh[:,:,int((nc-1)/2),:-1,:] + mesh[:,:,int((nc-1)/2),1:,:])/2 # num_nodes, nt, ns - 1, 3

            chord_spanwise = csdl.norm(surface_TE - surface_LE, axes=(3,)) # num_nodes, nt, ns - 1

            planform_area = csdl.sum(chord_spanwise*avg_panel_width_spanwise, axes=(2,))
            mesh_dict[key]['planform_area'] = planform_area



    elif mode == 'unstructured':
        pass

    return mesh_dict