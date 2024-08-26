import numpy as np
from VortexAD import AIRFOIL_PATH
 
def gen_rotor_mesh(nc, chord_array, span, r_inner, airfoil='naca0012', twist=0., plot_mesh=False):
    if airfoil not in ['naca0012']:
        raise ImportError('Airfoil not added yet.')
    else:
        loaded_data = np.loadtxt(str(AIRFOIL_PATH) + '/' + 'naca0012.txt', skiprows=1) # NEED TO USE SELIG FILE FORMAT
 
    zero_ind = np.where(loaded_data[:,0] == loaded_data[:,0].min())[0][0] # location of 0 in the chord data
 
    index_linspace = np.linspace(0,1,nc)
    num_pts = loaded_data.shape[0]
    airfoil_data = {
        'x': loaded_data[:,0],
        'z': loaded_data[:,1] * -1., # flips data so that we go from lower surface to upper surface
        'indices': np.arange(num_pts)
    }
    # airfoil_data['x'][:zero_ind]
    ns = len(chord_array)
    # origin at the wing LE center
    mesh = np.zeros((2*nc-1, ns, 3))
    # mesh = np.zeros((nc, ns, 3))
 
    # normalized chord and thickness data
    c_mesh_interp = np.linspace(-1, 1, nc)
    zero_ind_c = np.where(c_mesh_interp == np.abs(c_mesh_interp).min())[0][0] # location of 0 in the chord data
 
    c_mesh_interp_lo = np.linspace(0, zero_ind, nc)
    c_mesh_interp_hi = np.linspace(zero_ind, num_pts-1, nc)[1:]
    c_mesh_interp = np.concatenate((c_mesh_interp_lo, c_mesh_interp_hi), axis=0)
 
    thickness_interp = np.interp(c_mesh_interp, airfoil_data['indices'], airfoil_data['z'])
    chord_interp = np.interp(c_mesh_interp, airfoil_data['indices'], airfoil_data['x'])
 
    # c_mesh_interp[:zero_ind_c] *= -1.
 
    span_array = np.linspace(r_inner, 1., ns)*span
    for i, y in enumerate(span_array):
        mesh[:,i,0] = chord_interp * chord_array[i]
        mesh[:,i,1] = y
        mesh[:,i,2] = thickness_interp * chord_array[i]
    
    mesh[0,:,:] = (mesh[0,:,:] + mesh[-1,:,:])/2.
    mesh[-1,:,:] = mesh[0,:,:]
 
    rot_mat = np.zeros((3,3))
    angle = np.deg2rad(twist)
    rot_mat[1,1] = 1.
    rot_mat[0,0] = rot_mat[2,2] = np.cos(angle)
    rot_mat[0,2] = np.sin(angle)
    rot_mat[2,0] = -np.sin(angle)
 
    blade_1 = np.einsum('kl,ijk->ijl', rot_mat, mesh)
 
    blade_2 = mesh.copy()
    blade_2[:,:,0] *= -1
    blade_2[:,:,1] *= -1
 
    if plot_mesh:
 
        import matplotlib.pyplot as plt
        fig = plt.figure()
        # plt.plot(airfoil_data['x'], airfoil_data['z'], 'v', label='upper')
        plt.plot(airfoil_data['x'], airfoil_data['z'], 'k', label='Airfoil data')
        plt.plot(chord_interp, thickness_interp, 'b*', label='Interpolation')
 
 
        plt.axis('equal')
 
        plt.legend()
        plt.show()
        
        import pyvista as pv
        p = pv.Plotter()
        pv_mesh = pv.wrap(blade_1.reshape((ns*int(2*nc-1),3)))
        p.add_mesh(pv_mesh, color='black')
        pv_mesh = pv.wrap(blade_2.reshape((ns*int(2*nc-1),3)))
        p.add_mesh(pv_mesh, color='black')
        p.show_axes()
        p.show()
 
 
    return [blade_1, blade_2]