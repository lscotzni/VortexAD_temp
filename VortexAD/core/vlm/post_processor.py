import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.vlm.compute_net_circulation import compute_net_circulation

def post_processor(num_nodes, mesh_dict, gamma):
    output_dict = {}

    net_gamma_dict = compute_net_circulation(num_nodes, mesh_dict, gamma)
    for key in mesh_dict.keys():
        sub_dict = {}
        sub_dict['net_gamma'] = net_gamma_dict[key]
        output_dict[key] = sub_dict


    asdf = csdl.get_current_recorder()
    asdf.print_graph_structure()
    asdf.visualize_graph()
    1


    



'''
Post-processing:
- compute net gamma based on bordering vortex panels
- compute induced velocities at the force evaluation points?
- compute lift and drag and the like
'''