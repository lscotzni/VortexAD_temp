import numpy as np
import csdl_alpha as csdl
import pickle
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh, gen_panel_mesh_new
from VortexAD import steady_panel_solver

# plotting functions
import matplotlib.pyplot as plt
from VortexAD.utils.plot import plot_pressure_distribution

# setting up inputs
b = 10.
c = 1.
num_nodes = 1
alpha_deg = 0.
alpha = np.deg2rad(alpha_deg) # aoa
mach = 0.15
sos = 340.3
Vx = sos*mach
V_inf = np.array([-Vx, 0., 0.])

# coarse and fine grids
ns, nc = 21, 31

mesh_orig = gen_panel_mesh_new(nc, ns, c, b,  frame='default', plot_mesh=False) # uneven chordwise spacing

mesh = np.zeros((num_nodes,) + mesh_orig.shape)
for i in range(num_nodes):
    mesh[i,:] = mesh_orig

V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocity = np.zeros_like(mesh)
for i in range(num_nodes):
    mesh_velocity[i,:] = V_inf_rot

recorder = csdl.Recorder(inline=False)
recorder.start()

# fine grid variables
mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_velocity)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocity]

# running original solver with fine grid
output_dict, mesh_dict, mu, sigma = steady_panel_solver(
    mesh_list, 
    mesh_velocity_list
)
wake_mesh_dict = output_dict['wake_dict']

panel_corners = mesh_dict['surface_0']['panel_corners']
panel_center = mesh_dict['surface_0']['panel_center']
panel_x_dir = mesh_dict['surface_0']['panel_x_dir']
panel_y_dir = mesh_dict['surface_0']['panel_y_dir']
panel_normal = mesh_dict['surface_0']['panel_normal']

S = mesh_dict['surface_0']['S']
SL = mesh_dict['surface_0']['SL']
SM = mesh_dict['surface_0']['SM']

panel_corners_wake = wake_mesh_dict['surface_0']['panel_corners']

Cp = output_dict['surface_0']['Cp']

inputs = [
    mesh,
    mesh_velocity
]

outputs = [
    Cp,
    mu,
    sigma,
    panel_corners,
    panel_center,
    panel_x_dir,
    panel_y_dir,
    panel_normal,
    S,
    SL,
    SM,
    panel_corners_wake
]

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocity], # list of outputs (put in csdl variable)
    additional_outputs=outputs, # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh = jax_sim[mesh]
mesh_velocity = jax_sim[mesh_velocity]
Cp = jax_sim[Cp]
mu = jax_sim[mu]
sigma = jax_sim[sigma]
panel_corners = jax_sim[panel_corners]
panel_center = jax_sim[panel_center]
panel_x_dir = jax_sim[panel_x_dir]
panel_y_dir = jax_sim[panel_y_dir]
panel_normal = jax_sim[panel_normal]
S = jax_sim[S]
SL = jax_sim[SL]
SM = jax_sim[SM]
panel_corners_wake = jax_sim[panel_corners_wake]

data_mesh_dict = {
    'surface_0':{
        'panel_corners': panel_corners,
        'panel_center': panel_center,
        'panel_x_dir': panel_x_dir,
        'panel_y_dir': panel_y_dir,
        'panel_normal': panel_normal,
        'S': S,
        'SL': SL,
        'SM': SM,
        'nc': 2*nc-1,
        'ns': ns,
        'num_panels': (ns-1)*2*(nc-1),
    }
}

data_wake_mesh_dict = {
    'surface_0': {
        'panel_corners': panel_corners_wake,
        'nc': wake_mesh_dict['surface_0']['nc'],
        'ns': wake_mesh_dict['surface_0']['ns'],
        'num_panels': wake_mesh_dict['surface_0']['num_panels'],
    }
}

output_dict = {
    'mesh_dict': data_mesh_dict,
    'wake_mesh_dict': data_wake_mesh_dict,
    'mu': mu,
    'sigma': sigma,
    'Cp': Cp,
    'mesh': mesh,
    'mesh_velocity': mesh_velocity,
}

with open('superposition_data.pickle', 'wb') as file:
    pickle.dump(output_dict, file)

if True:
    plot_pressure_distribution([mesh], [Cp], interactive=True, top_view=False)


