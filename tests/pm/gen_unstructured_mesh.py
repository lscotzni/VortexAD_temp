import csdl_alpha as csdl
import numpy as np 
from VortexAD.core.geometry.gen_panel_mesh import gen_panel_mesh

b = 10
c = 1
# ns = 11
ns = 11
nc = 11

alpha = np.deg2rad(0.) # aoa

mach = 0.25
sos = 340.3
V_inf = np.array([sos*mach, 0., 0.])
nt = 5
num_nodes = 1

points_orig, connectivity = gen_panel_mesh(nc, ns, c, b, frame='default', unstructured=True, plot_mesh=False)