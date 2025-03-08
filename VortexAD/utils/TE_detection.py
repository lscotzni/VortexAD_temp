import numpy as np

def TE_detection(points, cells, edges2cells, threshold_theta=75, use_caddee=False):

    # getting edge vectors of panel
    # points are ordered s.t. outward normal
    p1_ind = cells[:,0]
    p2_ind = cells[:,1]
    p3_ind = cells[:,2]
    v1 = points[p2_ind,:] - points[p1_ind,:]
    v2 = points[p3_ind,:] - points[p2_ind,:]
    cell_normal = np.cross(v1, v2, axis=1)
    cell_normal_norm = np.linalg.norm(cell_normal, axis=1) # cell normal norm

    # cosine of TE threshold angle
    theta_t = np.deg2rad(threshold_theta)
    threshold_cos = np.cos(theta_t)

    gcs_scaler = 1
    if use_caddee:
        gcs_scaler = -1

    '''
    The loop below finds the TE edges based on 3 criteria:
    - CRITERIA 1: norm of dot product between normals is below some threshold
        - sharpness of angle between two panels is the main criteria
    - CRITERIA 2: at least one of the vectors points downstream
        - NOTE: REFERENCE FRAMES MATTER HERE (thinking about CADDEE)
        - This leaves out leading edge panels
    - CRITERIA 3: flow turns away from surface
        - this will ignore crevices that "recirculate" flow into the body
        - we want the flow to turn away from the body for wake-shedding
    '''

    upper_TE_cells = []
    lower_TE_cells = []
    TE_edges = []
    node_TE_indices = []
    for edge in edges2cells.keys():
        cell_pairs = edges2cells[edge]
        if len(cell_pairs) < 2:
            continue
        cell_1, cell_2 = cell_pairs[0], cell_pairs[1]
        n1 = cell_normal[cell_1]/cell_normal_norm[cell_1]
        n2 = cell_normal[cell_2]/cell_normal_norm[cell_2]

        # CRITERIA 1
        edge_angle_cos = np.dot(n1, n2)
        if edge_angle_cos > threshold_cos:
            continue
        
        # CRITERIA 2:
        if gcs_scaler*n1[0] <= 0:
            if gcs_scaler*n2[0] <= 0:
                continue

        # CRITERIA 3:

        n_cross = np.cross(n1, n2)
        # c3 = np.dot(l, n_cross)
        # if c3 < 0:
        #     continue

        # finding upper and lower cells
        # upper: other vertex is above the edge
        #   - normal vector points up
        # lower: other vertex is below the edge
        #   - normal vector points down

        if n1[2] > 0:
            upper_TE_cells.append(int(cell_1))
            lower_TE_cells.append(int(cell_2))
        # elif n2[2] > 0:
        else:
            upper_TE_cells.append(int(cell_2))
            lower_TE_cells.append(int(cell_1))

        TE_edges.append(edge)
        node_TE_indices.extend(edge)

    node_TE_indices = list(set(node_TE_indices))

    return upper_TE_cells, lower_TE_cells, TE_edges, node_TE_indices