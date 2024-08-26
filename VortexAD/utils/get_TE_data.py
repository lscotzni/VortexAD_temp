import numpy as np

def get_TE_data(points, cells, cell_adjacency, edges2cells):

    TE_x = 1.
    TE_nodes_ordered = np.where(points[:,0] == TE_x)[0] # nodes on the trailing edge

    TE_points = points[TE_nodes_ordered][:,1]

    TE_pt_ind_ordered = TE_points.argsort()
    TE_nodes = TE_nodes_ordered[TE_pt_ind_ordered]
    
    # finding TE edges
    edges = list(edges2cells.keys())
    TE_edges = []
    for i, val in enumerate(TE_nodes[:-1]):
        for j in TE_nodes[(i+1):]:
            edge = (val,j)
            if edge in edges:
                TE_edges.append(edge)
            elif edge[::-1] in edges:
                TE_edges.append(edge[::-1])

    TE_upper_cells, TE_lower_cells = [], []
    for i, edge in enumerate(TE_edges):
        TE_cells = edges2cells[edge] # always 2 elements
        cell_1, cell_2 = TE_cells[0], TE_cells[1]
        node_1, node_2 = edge[0], edge[1]

        cell_1_nodes = list(cells[cell_1])
        cell_1_nodes.remove(node_1)
        cell_1_nodes.remove(node_2)

        cell_2_nodes = list(cells[cell_2])
        cell_2_nodes.remove(node_1)
        cell_2_nodes.remove(node_2)

        p3_1 = points[cell_1_nodes]
        p3_1_z = p3_1[0,-1]

        p3_2 = points[cell_2_nodes]
        p3_2_z = p3_2[0,-1]


        if p3_1_z > p3_2_z:
            TE_upper_cells.append(cell_1)
            TE_lower_cells.append(cell_2)
        elif p3_1_z < p3_2_z:
            TE_lower_cells.append(cell_1)
            TE_upper_cells.append(cell_2)


    return TE_upper_cells, TE_lower_cells, TE_nodes