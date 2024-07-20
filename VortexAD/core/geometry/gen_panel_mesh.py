import numpy as np
from VortexAD import AIRFOIL_PATH

from VortexAD.core.geometry.gen_gmsh_unstructured_mesh import gen_gmsh_unstructured_mesh, convert_to_unstructured

def gen_panel_mesh_new(nc, ns, chord, span, airfoil='naca0012', frame='default', plot_mesh=False):
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

    span_array = np.linspace(-span/2, span/2, ns)
    for i, y in enumerate(span_array):
        mesh[:,i,0] = chord_interp * chord
        mesh[:,i,1] = y
        mesh[:,i,2] = thickness_interp * chord
    
    mesh[0,:,:] = (mesh[0,:,:] + mesh[-1,:,:])/2.
    mesh[-1,:,:] = mesh[0,:,:]

    # airfoil_data['x'][:zero_ind] *= -1

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
        # plt.plot(airfoil_data['x'], airfoil_data['z'], 'v', label='upper')
        plt.plot(airfoil_data['x'], airfoil_data['z'], 'k', label='Airfoil data')
        plt.plot(chord_interp, thickness_interp, 'b*', label='Interpolation')


        plt.axis('equal')
        if frame == 'caddee':
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

        plt.legend()
        plt.show()
        
        import pyvista as pv
        p = pv.Plotter()
        pv_mesh = pv.wrap(mesh.reshape((ns*int(2*nc-1),3)))
        p.add_mesh(pv_mesh, color='black')
        p.show()

    return mesh


def gen_panel_mesh(nc, ns, chord, span, span_spacing='linear', airfoil='naca0012', frame='default', unstructured=False, plot_mesh=False):
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

    airfoil_data['z'][-1] = (airfoil_data['z'][0]+airfoil_data['z'][-1]) / 2.
    airfoil_data['z'][0] = airfoil_data['z'][-1] 

    # origin at the wing LE center
    mesh = np.zeros((2*nc-1, ns, 3))

    # normalized chord and thickness data
    c_mesh_interp = np.linspace(-1, 1, 2*nc - 1)
    zero_ind_c = np.where(c_mesh_interp == np.abs(c_mesh_interp).min())[0][0] # location of 0 in the chord data

    thickness_interp = np.interp(c_mesh_interp, airfoil_data['x'], airfoil_data['z'])

    c_mesh_interp[:zero_ind_c] *= -1.

    # if span_spacing == 'linear':
    span_array = np.linspace(-span/2, span/2, ns)
    # print('old span:', span_array)
    if span_spacing == 'cosine':
        theta = np.linspace(0, np.pi, ns)
        cos = np.cos(theta)
        span_array = -cos*span/2
        # print('new span:', span_array)
        # exit()

    if not unstructured:
        for i, y in enumerate(span_array):
            mesh[:,i,0] = c_mesh_interp * chord
            mesh[:,i,1] = y
            mesh[:,i,2] = thickness_interp * chord
        
        mesh[0,:,:] = (mesh[0,:,:] + mesh[-1,:,:])/2.
        mesh[-1,:,:] = mesh[0,:,:]

    elif unstructured:
        mesh = gen_gmsh_unstructured_mesh(span_array, thickness_interp, c_mesh_interp*chord, name=airfoil+'_mesh')
        points, connectivity = convert_to_unstructured(mesh)

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
        plt.plot(airfoil_data['x'], airfoil_data['z'], 'k', label='Airfoil data')
        plt.plot(c_mesh_interp, thickness_interp, 'b*', label='Interpolation')
        plt.xlabel('Normalized chord (x/c)')
        plt.ylabel('Thickness (z)')
        
        plt.axis('equal')
        if frame == 'caddee':
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

        plt.legend()
        plt.grid()
        plt.ylim([-0.1, 0.1])
        plt.show()
        if not unstructured:
            import pyvista as pv
            p = pv.Plotter()
            pv_mesh = pv.wrap(mesh.reshape((ns*(2*nc-1),3)))
            p.add_mesh(pv_mesh, color='black')
            p.show()

    if not unstructured:
        return mesh
    
    else:
        return points, connectivity