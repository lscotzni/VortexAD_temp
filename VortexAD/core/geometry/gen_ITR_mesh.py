import numpy as np 
from VortexAD import AIRFOIL_PATH

def gen_ITR_mesh(nc=21, ns=11, B=4, chord_spacing='uniform', span_spacing='uniform', plot_mesh=False):
    # if chord_spacing not in ['uniform', 'cosine']:
    #     raise ValueError('Invalid chord spacing. Options are uniform or cosine')
    if span_spacing not in ['uniform', 'cosine']:
        raise ValueError('Invalid chord spacing. Options are uniform or cosine')
    
    # ============ ITR parameters ============ 
    R = 0.1588 # radius
    c_R = 0.2
    c = R*c_R
    r = [0.2, 1.] # inner and outer nondimensional radii
    B = B # number of blades
    th_tip = 6.9 # tip twist angle
    
    spar_loc = 0.25 # nondimensional spar location along the chord

    nondim_span_array = np.linspace(r[0], r[1], ns)
    twist_array = th_tip/nondim_span_array
    # if span_spacing == 'cosine':
    #     span_relative = (span_array - r[0])/(r[1]-r[0])
    #     theta_array = np.pi*span_relative
        

        
    
    # load NACA0012 airfoil data
    loaded_data = np.loadtxt(str(AIRFOIL_PATH) + '/' + 'naca0012.txt', skiprows=1) # NEED TO USE SELIG FILE FORMAT
    num_airfoil_pts = loaded_data.shape[0]
    zero_ind = np.where(loaded_data[:,0] == loaded_data[:,0].min())[0][0] # location of 0 in the chord data

    airfoil_data = {
        'x': loaded_data[:,0],
        'z': loaded_data[:,1],
        'indices': np.arange(num_airfoil_pts)
    }

    if chord_spacing == 'uniform':
        airfoil_data['x'][:zero_ind] *= -1
        chord_interp = np.linspace(-1, 1, 2*nc - 1)
        zero_ind_c = np.where(chord_interp == np.abs(chord_interp).min())[0][0] # location of 0 in the chord data

        thickness_interp = np.interp(chord_interp, airfoil_data['x'], airfoil_data['z'])

        chord_interp[:zero_ind_c] *= -1.
        airfoil_data['x'][:zero_ind] *= -1
    elif chord_spacing == 'cosine':

        c_mesh_interp = np.linspace(-1,1,nc)
        zero_ind_c = np.where(c_mesh_interp == np.abs(c_mesh_interp).min())[0][0]

        c_mesh_interp_lo = np.linspace(0, zero_ind, nc)
        c_mesh_interp_hi = np.linspace(zero_ind, num_airfoil_pts-1, nc)[1:]
        c_mesh_interp = np.concatenate((c_mesh_interp_lo, c_mesh_interp_hi), axis=0)

        thickness_interp = np.interp(c_mesh_interp, airfoil_data['indices'], airfoil_data['z'])
        chord_interp = np.interp(c_mesh_interp, airfoil_data['indices'], airfoil_data['x'])
    else:
        raise TypeError('invalid chord spacing. Options are default or cosine')
    
    # Untwisted mesh
    mesh_not_rotated = np.zeros((2*nc-1, ns, 3))
    for i in range(ns):
        mesh_not_rotated[:,i,0] = chord_interp * c
        mesh_not_rotated[:,i,1] = nondim_span_array[i] * R
        mesh_not_rotated[:,i,2] = thickness_interp * c

    # Twisted mesh
    rot_mat = np.zeros((3,3))
    rot_mat[1,1] = 1.
    rot_point = np.array([spar_loc, 0., 0.])*c
    rot_point_exp = np.einsum(
        'i,j->ij',
        np.ones((2*nc-1,)),
        rot_point
    )

    mesh = np.zeros((2*nc-1, ns, 3))
    for i in range(ns):
        nondim_span = nondim_span_array[i]
        twist = np.deg2rad(twist_array[i]) # this is like a "negative" rotation

        rot_mat[0,0] = rot_mat[2,2] = np.cos(twist)
        rot_mat[2,0] = -np.sin(twist)
        rot_mat[0,2] = np.sin(twist)

        origin_shift =  mesh_not_rotated[:,i,:] - rot_point_exp
        rotated_pts_shift = np.einsum(
            'ij,aj->ai',
            rot_mat,
            origin_shift,
        )
        # rotated_pts = rotated_pts_shift + rot_point_exp
        rotated_pts = rotated_pts_shift 

        mesh[:,i,:] = rotated_pts
    mesh = mesh[::-1,:,:]
    # mesh = mesh[:,::-1,:]
    mesh_list = [mesh]
    dtheta_blade = 2*np.pi/B
    rot_mat = np.zeros((3,3))
    rot_mat[2,2] = 1.

    for i in range(1, B):
        mesh_copy = mesh.copy()

        rot_angle = dtheta_blade*i
        rot_mat[0,0] = rot_mat[1,1] = np.cos(rot_angle)
        rot_mat[1,0] = np.sin(rot_angle)
        rot_mat[0,1] = -np.sin(rot_angle)

        mesh_copy = np.einsum('ij,abj->abi', rot_mat, mesh_copy)
        mesh_list.append(mesh_copy)

    if plot_mesh:

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # # plt.plot(airfoil_data['x'], airfoil_data['z'], 'v', label='upper')
        # plt.plot(airfoil_data['x'], airfoil_data['z'], 'k', label='Airfoil data')
        # plt.plot(chord_interp, thickness_interp, 'b*', label='Interpolation')

        # plt.axis('equal')

        # plt.legend()
        # plt.show()

        # corners = np.stack((xcorn, ycorn, zcorn))
        # corners = corners.transpose()

        
        # grid = pv.ExplicitStructuredGrid(dims, corners)
        # grid = grid.compute_connectivity()
        # grid.plot(show_edges=True)
        
        import pyvista as pv
        p = pv.Plotter()
        dims = np.asarray((2*nc-1, ns, 3)) + 1
        for blade_mesh in mesh_list:
            pv_mesh = pv.wrap(blade_mesh.reshape((ns*int(2*nc-1),3)))
            p.add_mesh(pv_mesh, color='black')
            # grid = pv.ExplicitStructuredGrid(dims, blade_mesh)
            # grid = grid.compute_connectivity()
            # grid.plot(show_edges=True)
        p.add_axes_at_origin(line_width=0.25)
        p.set_scale(xscale=10., yscale=10., zscale=10.)
        p.show()

    return mesh_list