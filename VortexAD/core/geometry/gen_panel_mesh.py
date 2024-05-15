import numpy as np
from VortexAD import AIRFOIL_PATH

def gen_panel_mesh(nc, ns, chord, span, airfoil='naca0012', frame='default', plot_mesh=True):
    if airfoil not in ['naca0012']:
        raise ImportError('Airfoil not added yet.')
    else:
        loaded_data = np.loadtxt(str(AIRFOIL_PATH) + '/' + 'naca0012.txt', skiprows=1) # NEED TO USE SELIG FILE FORMAT

    zero_ind = np.where(loaded_data[:,0] == loaded_data[:,0].min())[0][0] # location of 0 in the chord data

    airfoil_data = {
        'x': loaded_data[:,0],
        'z': loaded_data[:,1] * -1. # flips data so that we go from lower surface to upper surface
    }
    airfoil_data['x'][:zero_ind] *= -1

    # origin at the wing LE center
    mesh = np.zeros((2*nc-1, ns, 3))

    # normalized chord and thickness data
    c_mesh_interp = np.linspace(-1, 1, 2*nc - 1)
    zero_ind_c = np.where(c_mesh_interp == np.abs(c_mesh_interp).min())[0][0] # location of 0 in the chord data

    thickness_interp = np.interp(c_mesh_interp, airfoil_data['x'], airfoil_data['z'])

    c_mesh_interp[:zero_ind_c] *= -1.

    span_array = np.linspace(-span/2, span/2, ns)
    for i, y in enumerate(span_array):
        mesh[:,i,0] = c_mesh_interp * chord
        mesh[:,i,1] = y
        mesh[:,i,2] = thickness_interp * chord

    airfoil_data['x'][:zero_ind] *= -1

    if frame == 'caddee':
        mesh[:,:,0] *= -1.
        mesh[:,:,2] *= -1.

        airfoil_data['x'] *= -1. 
        airfoil_data['z'] *= -1.
        c_mesh_interp *= -1.
        thickness_interp *= -1.



    if plot_mesh:

        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(airfoil_data['x'], airfoil_data['z'], label='upper')
        plt.plot(c_mesh_interp, thickness_interp, label='upper interpolated')
        plt.axis('equal')
        if frame == 'caddee':
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

        plt.legend()
        plt.show()
        
        import pyvista as pv
        p = pv.Plotter()
        pv_mesh = pv.wrap(mesh.reshape((ns*(2*nc-1),3)))
        p.add_mesh(pv_mesh, color='black')
        p.show()

    return mesh