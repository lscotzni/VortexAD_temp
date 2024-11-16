import numpy as np
from VortexAD import AIRFOIL_PATH

def gen_onera_m6_mesh(nc, ns, chord_spacing='default', span_spacing='default', plot_mesh=False):
    LE_sweep = 30.
    root_chord = 0.8105
    tip_chord = 0.4559
    semi_span = 1.1963
    # loading airfoil data for upper surface
    upper_surf = np.loadtxt(str(AIRFOIL_PATH) + '/onera_m6_airfoil.dat')
    lower_surf = upper_surf.copy()
    lower_surf[:,1] *= -1
    upper_surf_pts = upper_surf.shape[0]
    num_airfoil_pts = int(2*(upper_surf_pts) - 1)

    airfoil_data_array = np.zeros((num_airfoil_pts,2))

    airfoil_data_array[:upper_surf_pts,0] = lower_surf[:,0][::-1]
    airfoil_data_array[:upper_surf_pts,1] = lower_surf[:,1][::-1]
    airfoil_data_array[(upper_surf_pts-1):,:] = upper_surf

    zero_ind = np.where(airfoil_data_array[:,0] == airfoil_data_array[:,0].min())[0][0]

    airfoil_data = {
        'x': airfoil_data_array[:,0],
        'z': airfoil_data_array[:,1],
        'indices': np.arange(num_airfoil_pts)
    }

    mesh = np.zeros((2*nc-1, ns, 3))

    if chord_spacing == 'default':
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

    span_array = np.linspace(-semi_span, semi_span, ns)
    half_span_ind = int((ns+1)/2)
    chord_dist = np.zeros_like(span_array)
    half_chord_dist = np.linspace(tip_chord, root_chord, half_span_ind)
    chord_dist[:half_span_ind] = half_chord_dist
    chord_dist[(half_span_ind-1):] = half_chord_dist[::-1]

    sweep_delta = np.abs(span_array * np.tan(LE_sweep*np.pi/180))

    for i, y in enumerate(span_array):
        mesh[:,i,0] = chord_interp * chord_dist[i] + sweep_delta[i]
        mesh[:,i,1] = y
        mesh[:,i,2] = thickness_interp * chord_dist[i]
    
    mesh[0,:,:] = (mesh[0,:,:] + mesh[-1,:,:])/2.
    mesh[-1,:,:] = mesh[0,:,:]

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
        pv_mesh = pv.wrap(mesh.reshape((ns*int(2*nc-1),3)))
        p.add_mesh(pv_mesh, color='black')
        p.show()

    return mesh