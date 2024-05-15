import csdl_alpha as csdl 

def initialize_unsteady_wake(mesh_dict, panel_fraction=0.25):
    surface_names = list(mesh_dict.keys())

    wake_mesh_dict = {}

    for i, surface_name in enumerate(surface_names):

        surf_wake_mesh_dict = {}

        surface_mesh = mesh_dict[surface_name]['mesh']
        mesh_velocity = mesh_dict[surface_name]['nodal_velocity']

        nt = surface_mesh.shape[1]

        TE = (surface_mesh[:,:,0,:,:] + surface_mesh[:,:,-1,:,:])/2.
        TE_vel = (mesh_velocity[:,:,0,:,:] + mesh_velocity[:,:,-1,:,:])/2.
        init_wake_pos = TE + panel_fraction*TE_vel*nt

        wake_mesh = csdl.Variable(shape=surface_mesh.shape[:2] + (nt+1,) + surface_mesh.shape[3:], value=0.)

        wake_mesh = wake_mesh.set(csdl.slice[:,:,0,:,:], value = TE)
        wake_mesh = wake_mesh.set(csdl.slice[:,:,1,:,:], value = init_wake_pos)
        
        surf_wake_mesh_dict['mesh'] = wake_mesh
        wake_mesh_dict[surface_name] = surf_wake_mesh_dict

    return wake_mesh_dict

