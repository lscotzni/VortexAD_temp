import numpy as np 
import csdl_alpha as csdl 

from VortexAD.core.geometry.gen_ITR_mesh import gen_ITR_mesh
from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver

from VortexAD.utils.plot import plot_wireframe, plot_transient_pressure_distribution

# ============ blade operating conditions ============
RPM = 5500
dt = 0.00025
# RPM = 10
# dt = 0.2
omega = RPM*2*np.pi/60. # ~ 576 rad/sec or 92 rev/sec 
omega_vec = np.array([0., 0., omega])
nt = 60

# ============ blade discretization and meshes ============
# the meshing function is specific to ITR, so all of the parameters are in the function
nc, ns = 11, 21
blade_mesh_list = gen_ITR_mesh(nc=nc, ns=ns, B=1, plot_mesh=True)
num_blades = len(blade_mesh_list)

blade_meshes = np.zeros((num_blades,2*nc-1, ns, 3))
for i in range(num_blades):
    blade_meshes[i,:,:,:] = blade_mesh_list[i]

# ============ computing blade position and collocation velocity in time ============
blade_mesh_array = np.zeros((1,nt,num_blades, 2*nc-1, ns, 3)) # nn, nt, num_blades, nc, ns, 3
blade_mesh_velocity_array = np.zeros_like(blade_mesh_array) # freestream, set to zero
blade_mesh_velocity_array[:,:,:,:,:,2] = 1.
blade_mesh_coll_vel_array = np.zeros((1,nt,num_blades,2*nc-2, ns-1,3))

rot_mat = np.zeros((3,3))
rot_mat[2,2] = 1
for i in range(nt):
    theta = dt*omega*i
    rot_mat[0,0] = rot_mat[1,1] = np.cos(theta)
    rot_mat[1,0] = -np.sin(theta)
    rot_mat[0,1] = np.sin(theta)
    rotated_blades = np.einsum('ij,abci->abcj', rot_mat, blade_meshes)
    # rotated_blades = np.cross(theta_vec, blade_meshes, axisa=0, axisb=3)
    blade_mesh_array[0,i,:,:,:,:] = rotated_blades

blade_coll_pt = (blade_mesh_array[:,:,:,:-1,:-1,:] + blade_mesh_array[:,:,:,1:,:-1,:] \
    + blade_mesh_array[:,:,:,1:,1:,:] + blade_mesh_array[:,:,:,:-1,1:,:]) / 4

for i in range(nt):
    coll_vel_timestep = np.cross(omega_vec, blade_coll_pt[0,i,:,:,:,:], axisa=0, axisb=3)
    blade_mesh_coll_vel_array[0,i,:,:,:,:] = coll_vel_timestep

recorder = csdl.Recorder(inline=True)
recorder.start()

blade_mesh_list = []
blade_mesh_vel_list = []
blade_coll_vel_list = []

for i in range(num_blades):
    blade_mesh_list.append(csdl.Variable(value=blade_mesh_array[:,:,i,:,:,:]))
    blade_mesh_vel_list.append(csdl.Variable(value=blade_mesh_velocity_array[:,:,i,:,:,:]))
    blade_coll_vel_list.append(csdl.Variable(value=blade_mesh_coll_vel_array[:,:,i,:,:,:]))
# exit()
output_dict, mesh_dict, wake_mesh_dict, mu, sigma, mu_wake = unsteady_panel_solver(
    blade_mesh_list,
    blade_mesh_vel_list,
    blade_coll_vel_list,
    dt=dt,
    free_wake=True
)

blade_wake_mesh_list = []
coll_points_list = []
blade_Cp_list = []

for i in range(num_blades):
    blade_wake_mesh_list.append(wake_mesh_dict[f'surface_{i}']['mesh'])
    coll_points_list.append(mesh_dict[f'surface_{i}']['panel_center'])
    blade_Cp_list.append(output_dict[f'surface_{i}']['Cp'])


inputs = []
inputs.extend(blade_mesh_list)

outputs = []
outputs.append(mu)
outputs.append(mu_wake)
outputs.extend(blade_wake_mesh_list)
outputs.extend(coll_points_list)
outputs.extend(blade_Cp_list)

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=inputs, # list of outputs (put in csdl variable)
    additional_outputs=outputs, # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh_list = [jax_sim[blade_mesh_list[i]] for i in range(num_blades)]
wake_mesh_list = [jax_sim[blade_wake_mesh_list[i]] for i in range(num_blades)]
Cp_list = [jax_sim[blade_Cp_list[i]] for i in range(num_blades)]
# mu_list = [jax_sim[mu]]
# mu_wake = [jax_sim[mu_wake]]

mu_list, mu_wake_list = [], []
mu, mu_wake = jax_sim[mu], jax_sim[mu_wake]
stop, start = 0, 0
stop_w, start_w = 0, 0
num_blade_panels = int((2*nc-2)*(ns-1))
num_blade_wake_panels = int((nt-1)*(ns-1))
for i in range(num_blades):
    stop += num_blade_panels
    mu_temp = mu[:,:,start:stop]
    mu_list.append(mu_temp)
    start += num_blade_panels

    stop_w += num_blade_wake_panels
    mu_wake_temp = mu_wake[:,:,start_w:stop_w]
    mu_wake_list.append(mu_wake_temp)
    start_w += num_blade_wake_panels


if True:
    from vedo import Axes

    axs = Axes(
        xrange=(-0.2, 0.2),
        yrange=(-0.2, 0.2),
        zrange=(0, 0.2),
    )
    # plot_wireframe(mesh, wake_mesh, mu.value, mu_wake.value, nt, interactive=False, backend='cv', name=f'wing_fw_{alpha_deg}')
    plot_wireframe(mesh_list, wake_mesh_list, mu_list, mu_wake_list, nt, interactive=False, backend='cv', name='ITR', axes=axs)

    # plot_transient_pressure_distribution(mesh_list[0], Cp_list[0], backend='cv', interactive=True, axes=axs)
