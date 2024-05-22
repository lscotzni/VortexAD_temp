import numpy as np 
import matplotlib.pyplot as plt 
import csdl_alpha as csdl

from VortexAD.core.vlm.vlm_solver import vlm_solver
from VortexAD import SAMPLE_GEOMETRY_PATH

wing_nodal_coordinate_path = str(SAMPLE_GEOMETRY_PATH) + '/vlm/c172_wing_nodal_coordinates.npy'
wing_nodal_velocity_path = str(SAMPLE_GEOMETRY_PATH) + '/vlm/c172_wing_nodal_velocities.npy'

wing_nodal_coordinates = np.load(wing_nodal_coordinate_path)
wing_nodal_velocity = np.load(wing_nodal_velocity_path)

tail_nodal_coordinate_path = str(SAMPLE_GEOMETRY_PATH) + '/vlm/c172_tail_nodal_coordinates.npy'
tail_nodal_velocity_path = str(SAMPLE_GEOMETRY_PATH) + '/vlm/c172_tail_nodal_velocities.npy'

tail_nodal_coordinates = np.load(tail_nodal_coordinate_path)
tail_nodal_velocity = np.load(tail_nodal_velocity_path)

mesh_list = [wing_nodal_coordinates, tail_nodal_coordinates]
mesh_velocity_list = [wing_nodal_velocity, tail_nodal_velocity]

recorder = csdl.Recorder(inline=True)
recorder.start()
output_vg = vlm_solver(mesh_list, mesh_velocity_list)
recorder.stop()

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

    # print('Surface panel forces (N): ', output_vg.surface_panel_forces[i].value)
    # print('Surface sectional center of pressure (m): ', output_vg.surface_sectional_cop[i].value)
    # print('Surface total center of pressure (m): ', output_vg.surface_cop[i].value)