import numpy as np 
import csdl_alpha as csdl

from VortexAD.utils.atan2_switch import atan2_switch

def pre_processor(mesh_dict, mode='structured', connectivity=None):
    surface_names = list(mesh_dict.keys())
    if mode == 'structured':
        for i, key in enumerate(surface_names): # looping over surface names
            mesh = mesh_dict[key]['mesh']
            mesh_shape = mesh.shape
            nc, ns = mesh_shape[-3], mesh_shape[-2]

            mesh_dict[key]['num_panels'] = (nc-1)*(ns-1)
            mesh_dict[key]['nc'] = nc
            mesh_dict[key]['ns'] = ns

            if i == 0:
                num_dim = len(mesh.shape) # last 3 dimensions are nc, ns, 3; ones before are either (nn,) or (nn, nt)
                base_slice = tuple([slice(None)] for j in range(num_dim - 3 + 1))

            # p1 =  mesh[base_slice + (slice(0, nc-1), slice(0, ns-1), slice(0,3))]
            # p2 =  mesh[base_slice + (slice(0, nc-1), slice(1, ns), slice(0,3))]
            # p3 =  mesh[base_slice + (slice(1, nc), slice(1, ns), slice(0,3))]
            # p4 =  mesh[base_slice + (slice(1, nc), slice(0, ns-1), slice(0,3))]

            p1 = mesh[:,:,:-1,:-1,:]
            p2 = mesh[:,:,:-1,1:,:]
            p3 = mesh[:,:,1:,1:,:]
            p4 = mesh[:,:,1:,:-1,:]

            panel_center = (p1 + p2 + p3 + p4)/4.
            mesh_dict[key]['panel_center'] = panel_center

            panel_corners = csdl.Variable(shape=(panel_center.shape[:-1] + (4,3)), value=0.)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,0,:], value=p1)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,1,:], value=p2)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,2,:], value=p3)
            panel_corners = panel_corners.set(csdl.slice[:,:,:,:,3,:], value=p4)
            mesh_dict[key]['panel_corners'] = panel_corners

            panel_diag_1 = p3-p1
            panel_diag_2 = p2-p4
            
            panel_normal_vec = csdl.cross(panel_diag_1, panel_diag_2, axis=4)
            panel_area = csdl.norm(panel_normal_vec, axes=(4,)) / 2.
            mesh_dict[key]['panel_area'] = panel_area

            panel_x_vec = p4 - p1
            panel_y_vec = p2 - p1

            panel_x_dir = panel_x_vec / csdl.expand((csdl.norm(panel_x_vec, axes=(4,))), panel_x_vec.shape, 'ijkl->ijkla')
            panel_y_dir = panel_y_vec / csdl.expand((csdl.norm(panel_y_vec, axes=(4,))), panel_y_vec.shape, 'ijkl->ijkla')
            panel_normal = panel_normal_vec / csdl.expand((csdl.norm(panel_normal_vec, axes=(4,))), panel_normal_vec.shape, 'ijkl->ijkla')

            mesh_dict[key]['panel_x_dir'] = panel_x_dir
            mesh_dict[key]['panel_y_dir'] = panel_y_dir
            mesh_dict[key]['panel_normal'] = panel_normal

            # COMPUTE DISTANCE FROM PANEL CENTER TO SEGMENT CENTERS

            pos_l = (p3+p4)/2.
            neg_l = (p1+p2)/2.
            pos_m = (p2+p3)/2.
            neg_m = (p1+p4)/2.

            pos_l_norm = csdl.norm(pos_l-panel_center, axes=(4,))
            neg_l_norm = csdl.norm(neg_l-panel_center, axes=(4,))
            pos_m_norm = csdl.norm(pos_m-panel_center, axes=(4,))
            neg_m_norm = csdl.norm(neg_m-panel_center, axes=(4,))

            mesh_dict[key]['panel_dl_norm'] = [pos_l_norm, neg_l_norm]
            mesh_dict[key]['panel_dm_norm'] = [pos_m_norm, neg_m_norm]

            delta_coll_point = csdl.Variable(panel_center.shape[:-1] + (4,2), value=0.)
            # shape is (num_nodes, nt, nc_panels, ns_panels, 4,2)
            # dimension of size 4 is dl backward, dl forward, dm backward, dm forward
            # dimension of size 2 is the projected deltas in the x or y direction
            # for panel i, the delta to adjacent panel j is taken as "j-i"
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,1:,:,0,0], value=csdl.sum((panel_center[:,:,:-1,:,:] - panel_center[:,:,1:,:,:] ) * panel_x_dir[:,:,1:,:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:-1,:,1,0], value=csdl.sum((panel_center[:,:,1:,:,:] - panel_center[:,:,:-1,:,:]) * panel_x_dir[:,:,:-1,:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,1:,2,1], value=csdl.sum((panel_center[:,:,:,:-1,:] - panel_center[:,:,:,1:,:]) * panel_y_dir[:,:,:,1:,:], axes=(4,)))
            delta_coll_point = delta_coll_point.set(csdl.slice[:,:,:,:-1,3,1], value=csdl.sum((panel_center[:,:,:,1:,:] - panel_center[:,:,:,:-1,:]) * panel_y_dir[:,:,:,:-1,:], axes=(4,)))

            mesh_dict[key]['delta_coll_point'] = delta_coll_point

            panel_center_mod = panel_center - panel_normal*0.0001
            mesh_dict[key]['panel_center'] = panel_center_mod

            # global unit vectors
            # +x points from tail to nose of aircraft
            # +y points to the left of the aircraft (facing from the front)
            # +z points down
            x_dir = np.array([1., 0., 0.,])
            y_dir = np.array([0., 1., 0.,])
            z_dir = np.array([0., 0., 1.,])

            Px_normal = csdl.tensordot(panel_normal, x_dir, axes=([4],[0]))
            Py_normal = csdl.tensordot(panel_normal, y_dir, axes=([4],[0]))
            Pz_normal = csdl.tensordot(panel_normal, z_dir, axes=([4],[0]))

            # theta_x = atan2_switch(Py_normal, Pz_normal, scale=100.) # roll angle
            # theta_y = atan2_switch(-Px_normal, Pz_normal, scale=100.) # pitch angle

            Px_x_dir = csdl.tensordot(panel_x_dir, x_dir, axes=([4],[0]))
            Py_x_dir = csdl.tensordot(panel_x_dir, y_dir, axes=([4],[0]))

            # theta_z = atan2_switch(Py_x_dir, Px_x_dir, scale=100.)

            dpij_global = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,3))))
            dpij_global = dpij_global.set(csdl.slice[:,:,:,:,0,:], value=p2[:,:,:,:,:]-p1[:,:,:,:,:])
            dpij_global = dpij_global.set(csdl.slice[:,:,:,:,1,:], value=p3[:,:,:,:,:]-p2[:,:,:,:,:])
            dpij_global = dpij_global.set(csdl.slice[:,:,:,:,2,:], value=p4[:,:,:,:,:]-p3[:,:,:,:,:])
            dpij_global = dpij_global.set(csdl.slice[:,:,:,:,3,:], value=p1[:,:,:,:,:]-p4[:,:,:,:,:])

            local_coord_vec = csdl.Variable(shape=(panel_center.shape[:-1] + (3,3)), value=0.) # last two indices are for 3 vectors, 3 dimensions
            local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,:,0,:], value=panel_x_dir)
            local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,:,1,:], value=panel_y_dir)
            local_coord_vec = local_coord_vec.set(csdl.slice[:,:,:,:,2,:], value=panel_normal)

            dpij_local = csdl.einsum(dpij_global, local_coord_vec, action='ijklma,ijklba->ijklmb')  # THIS IS CORRECT

            dpij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,2))))
            dpij = dpij.set(csdl.slice[:,:,:,:,0,:], value=dpij_local[:,:,:,:,0,:2])
            dpij = dpij.set(csdl.slice[:,:,:,:,1,:], value=dpij_local[:,:,:,:,1,:2])
            dpij = dpij.set(csdl.slice[:,:,:,:,2,:], value=dpij_local[:,:,:,:,2,:2])
            dpij = dpij.set(csdl.slice[:,:,:,:,3,:], value=dpij_local[:,:,:,:,3,:2])
            # dpij = dpij.set(csdl.slice[:,:,:,:,0,:], value=p2[:,:,:,:,:2]-p1[:,:,:,:,:2])
            # dpij = dpij.set(csdl.slice[:,:,:,:,1,:], value=p3[:,:,:,:,:2]-p2[:,:,:,:,:2])
            # dpij = dpij.set(csdl.slice[:,:,:,:,2,:], value=p4[:,:,:,:,:2]-p3[:,:,:,:,:2])
            # dpij = dpij.set(csdl.slice[:,:,:,:,3,:], value=p1[:,:,:,:,:2]-p4[:,:,:,:,:2])

            mesh_dict[key]['dpij'] = dpij

            dij = csdl.Variable(value=np.zeros((panel_center.shape[:-1] + (4,))))
            dij = dij.set(csdl.slice[:,:,:,:,0], value=csdl.norm(dpij[:,:,:,:,0,:], axes=(4,)))
            dij = dij.set(csdl.slice[:,:,:,:,1], value=csdl.norm(dpij[:,:,:,:,1,:], axes=(4,)))
            dij = dij.set(csdl.slice[:,:,:,:,2], value=csdl.norm(dpij[:,:,:,:,2,:], axes=(4,)))
            dij = dij.set(csdl.slice[:,:,:,:,3], value=csdl.norm(dpij[:,:,:,:,3,:], axes=(4,)))

            mesh_dict[key]['dij'] = dij

            mij = csdl.Variable(shape=dij.shape, value=0.)
            mij = mij.set(csdl.slice[:,:,:,:,0], value=(dpij[:,:,:,:,0,1])/(dpij[:,:,:,:,0,0]+1.e-12))
            mij = mij.set(csdl.slice[:,:,:,:,1], value=(dpij[:,:,:,:,1,1])/(dpij[:,:,:,:,1,0]+1.e-12))
            mij = mij.set(csdl.slice[:,:,:,:,2], value=(dpij[:,:,:,:,2,1])/(dpij[:,:,:,:,2,0]+1.e-12))
            mij = mij.set(csdl.slice[:,:,:,:,3], value=(dpij[:,:,:,:,3,1])/(dpij[:,:,:,:,3,0]+1.e-12))

            mesh_dict[key]['mij'] = mij

            nodal_vel = mesh_dict[key]['nodal_velocity']
            mesh_dict[key]['coll_point_velocity'] = (nodal_vel[:,:,:-1,:-1,:]+nodal_vel[:,:,:-1,1:,:]+nodal_vel[:,:,1:,1:,:]+nodal_vel[:,:,1:,:-1,:]) / 4.

            # computing planform area
            panel_width_spanwise = csdl.norm((mesh[:,:,:,1:,:] - mesh[:,:,:,:-1,:]), axes=(4,))
            avg_panel_width_spanwise = csdl.average(panel_width_spanwise, axes=(2,)) # num_nodes, nt, ns - 1
            surface_TE = (mesh[:,:,-1,:-1,:] + mesh[:,:,0,:-1,:] + mesh[:,:,-1,1:,:] + mesh[:,:,0,1:,:])/4
            surface_LE = (mesh[:,:,int((nc-1)/2),:-1,:] + mesh[:,:,int((nc-1)/2),1:,:])/2 # num_nodes, nt, ns - 1, 3

            chord_spanwise = csdl.norm(surface_TE - surface_LE, axes=(3,)) # num_nodes, nt, ns - 1

            planform_area = csdl.sum(chord_spanwise*avg_panel_width_spanwise, axes=(2,))
            mesh_dict[key]['planform_area'] = planform_area

    elif mode == 'unstructured':
        # use point list and connectivity
        for i, key in enumerate(surface_names): # looping over surface names
            mesh = mesh_dict[key]['mesh'] # num_nodes, nt, num_panels, 3
            mesh_shape = mesh.shape

            p1 = mesh[:,:,list(connectivity[:,0])]





    return mesh_dict