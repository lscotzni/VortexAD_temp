import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.geometry.gen_vlm_mesh import gen_vlm_mesh
from VortexAD.core.vlm.vlm_solver import vlm_solver

# flow parameters
frame = 'caddee'
vnv_scaler =  1.
num_nodes = 1
alpha = np.array([5.,]) * np.pi/180.
V_inf = np.array([-60, 0., 0.])
if frame == 'caddee':
    V_inf *= -1.
    vnv_scaler = -1.

# grid setup
ns = 11
nc = 3
b = 10
c = 1
# nc, ns = 11, 15

# generating mesh
mesh_orig = gen_vlm_mesh(ns, nc, b, c, frame=frame)
mesh = np.zeros((num_nodes,) + mesh_orig.shape)
for i in range(num_nodes):
    mesh[i,:,:,:] = mesh_orig

# setting up mesh velocity
V_rot_mat = np.zeros((3,3))
V_rot_mat[1,1] = 1.
V_rot_mat[0,0] = V_rot_mat[2,2] = np.cos(alpha)
V_rot_mat[2,0] = np.sin(alpha)
V_rot_mat[0,2] = -np.sin(alpha)
V_inf_rot = np.matmul(V_rot_mat, V_inf)

mesh_velocity = np.zeros_like(mesh)
mesh_velocity[:,:,:,0] = V_inf_rot[0]
mesh_velocity[:,:,:,2] = V_inf_rot[2]

# solver input setup
recorder = csdl.Recorder(inline=True)
recorder.start()
mesh = csdl.Variable(value=mesh)
mesh_velocity = csdl.Variable(value=mesh_velocity)
mesh_list = [mesh]
mesh_velocity_list = [mesh_velocity]

# alpha_ML = np.ones((num_nodes, ns-1)) * -5*np.pi/180.
# alpha_ML = None


output_vg = vlm_solver(mesh_list, mesh_velocity_list)
wing_CL = output_vg.surface_CL[0]

# deriv = csdl.derivative_utils.verify_derivatives(ofs = wing_CL, wrts=mesh)

recorder.stop()

# py_sim = csdl.experimental.PySimulator(
#         recorder=recorder,
#     )   
#     py_sim.check_totals(ofs=csdl.average(vlm_outputs.AIC_force_eval_pts), wrts=elevator_deflection)

from csdl_alpha.experimental import PySimulator

py_sim = PySimulator(
    recorder=recorder
)
py_sim.check_totals(ofs=wing_CL, wrts=mesh)

# recorder.print_graph_structure()
# recorder.visualize_graph(filename='ex1_oas_graph')

wing_CL = output_vg.surface_CL[0].value
wing_CDi = output_vg.surface_CDi[0].value

wing_CL_OAS = np.array([0.4426841725811703]) * vnv_scaler
wing_CDi_OAS = np.array([0.005878842561184834]) * vnv_scaler

CL_error = (wing_CL_OAS - wing_CL)/(wing_CL_OAS) * 100
CDi_error = (wing_CDi_OAS - wing_CDi)/(wing_CDi_OAS) * 100

print('======== ERROR PERCENTAGES (OAS - VortexAD) ========')
print(f'CL error (%): {CL_error}')
print(f'CDi error (%): {CDi_error}')

print('======  PRINTING TOTAL OUTPUTS ======')
print('Total force (N): ', output_vg.total_force.value)
print('Total Moment (Nm): ', output_vg.total_moment.value)
print('Total lift (N): ', output_vg.total_lift.value)
print('Total drag (N): ', output_vg.total_drag.value)

print('======  PRINTING OUTPUTS PER SURFACE ======')
for i in range(len(mesh_list)): # LOOPING THROUGH NUMBER OF SURFACES
    print('======  SURFACE 1 ======')
    print('Surface total force (N): ', output_vg.surface_force[i].value)
    print('Surface total moment (Nm): ', output_vg.surface_moment[i].value)
    print('Surface total lift (N): ', output_vg.surface_lift[i].value)
    print('Surface total drag (N): ', output_vg.surface_drag[i].value)
    print('Surface CL: ', output_vg.surface_CL[i].value)
    print('Surface CDi : ', output_vg.surface_CDi[i].value)

    print('Surface panel forces (N): ', output_vg.surface_panel_forces[i].value)
    print('Surface sectional center of pressure (m): ', output_vg.surface_sectional_cop[i].value)
    print('Surface total center of pressure (m): ', output_vg.surface_cop[i].value)

1