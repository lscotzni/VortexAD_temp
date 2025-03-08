import csdl_alpha as csdl
import numpy as np 
from numpy.linalg import eig, eigh, svd

from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh, gen_panel_mesh_new
from VortexAD import steady_panel_solver

# plotting functions
import matplotlib.pyplot as plt
from VortexAD.utils.plot import plot_pressure_distribution

b = 10.
c = 1.
ns = 11
nc = 21
num_nodes = 1

aoa = np.linspace(0, 10, 10)
alpha_deg = aoa[9]
alpha = np.deg2rad(alpha_deg) # aoa

mach = 0.15
sos = 340.3
Vx = sos*mach
V_inf = np.array([-Vx, 0., 0.])

mesh_orig = gen_panel_mesh(nc, ns, c, b, span_spacing='cosine',  frame='default', plot_mesh=False) # even chordwise spacing
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

mesh_velocities = np.zeros_like(mesh)
for i in range(num_nodes):
    mesh_velocities[i,:] = V_inf_rot

recorder = csdl.Recorder(inline=False)
recorder.start()

mesh = csdl.Variable(value=mesh)
mesh_velocities = csdl.Variable(value=mesh_velocities)

mesh_list = [mesh]
mesh_velocity_list = [mesh_velocities]

'''
inputs to the solver:
- list of meshes
- list of mesh velocities
- actuation velocities (if rotating bodies) -> not applicable for now
'''
output_dict, mesh_dict, mu, sigma = steady_panel_solver(
    mesh_list, 
    mesh_velocity_list
)

coll_points = mesh_dict['surface_0']['panel_center']
Cp = output_dict['surface_0']['Cp']
CL  = output_dict['surface_0']['CL']
CDi = output_dict['surface_0']['CDi']
AIC_mu = output_dict['AIC_mu']
AIC_sigma = output_dict['AIC_sigma']

recorder.stop()
jax_sim = csdl.experimental.JaxSimulator(
    recorder=recorder,
    additional_inputs=[mesh, mesh_velocities], # list of outputs (put in csdl variable)
    additional_outputs=[mu, sigma, coll_points, Cp, CL, CDi, AIC_mu, AIC_sigma], # list of outputs (put in csdl variable)
)
jax_sim.run()

mesh = jax_sim[mesh]
coll_points = jax_sim[coll_points]
Cp = jax_sim[Cp]    # panel pressure
CL = jax_sim[CL]
CDi = jax_sim[CDi]

# terms needed for ROM
mu = jax_sim[mu]
sigma = jax_sim[sigma]
AIC_mu = jax_sim[AIC_mu]
AIC_sigma = jax_sim[AIC_sigma]
'''
linear system is AIC_mu*mu + AIC_sigma*sigma = 0
RHS = -AIC_sigma*sigma
mu = AIC_mu^{-1}(RHS)
note that there is a num_nodes dimension, so a simple np.matmul or 
np.linalg.solve does NOT work. You need to remove num_nodes first
using something like A = A[0,:]
'''

if True:
    plot_pressure_distribution([mesh], [Cp], interactive=True, top_view=False)


# ## SVD and eigenvalue system reduction test w.o design variables
# from scipy.io import savemat
# re_shape = (400, 400)
# A = np.reshape(AIC_mu, re_shape)
# B = np.reshape(AIC_sigma, re_shape)

# nDOF = 400
# F = np.zeros((nDOF,1))
# F[-1] = 1

# GU = np.linalg.inv(A)@F
# np.savetxt('GU9.txt', GU, fmt='%.15f')

# savemat("panel.mat", {"A": A, "B": B})
# eig_val, eig_vec = eig(A,b=B)

# eVal, eVec = eigh(A, B)
# from numpy.linalg import eig

# data_export = np.column_stack((tau_interp, p_FWH_T, p_FWH_L, p_FWH))
# Cp_col = Cp.reshape(-1)
# np.savetxt('Cp10.txt', Cp_col, fmt='%.15f')