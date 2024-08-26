import numpy as np

def find_cell_adjacency(points, cells):
    '''
    points has a shape of (num_points, 3) where the 3 is the x,y,z coordinates
    cells has a shape of (num_cells, 3) where the 3 is the node indices of the cell
    '''
    num_pts = len(points)
    num_cells = len(cells)

    # Finding the edge corresponding to a cell (edge tuple is the dictionary key)
    edges2cells = {}
    for c, cell in enumerate(cells):
        cell = list(cell)
        cell.append(cell[0])
        edges = [(cell[i], cell[i+1]) for i in range(len(cell)-1)]

        for edge in edges:
            edge_rev = edge[::-1]
            if edge in edges2cells.keys():
                edges2cells[edge].append(c)
            elif edge_rev in edges2cells.keys():
                edges2cells[edge_rev].append(c)
            else:
                edges2cells[edge] = [c]

    # Finding neighboring cells (cell index is the dict key)
    cell_adjacency = {i: [] for i in range(num_cells)}
    for edge in edges2cells.keys():
        cell_pairs = edges2cells[edge]
        if len(cell_pairs) < 2:
            continue # means this is an edge on the border of the mesh
        cell_adjacency[cell_pairs[0]].append(cell_pairs[1])
        cell_adjacency[cell_pairs[1]].append(cell_pairs[0])

    cell_adjacency = np.array(list(cell_adjacency.values()))

    # # reordering points and cells to remove duplicate indices (TEMPORARY)
    # cells_new = np.zeros_like(cells)
    # new_points = []

    # rep_point_counter = 0
    # for i in range(num_pts):
    #     a = np.where(cells == i)
    #     num_rep = len(a[0])

    #     for j in range(num_rep):
    #         new_points.append(points[i])

    #         row, col = a[0][j], a[1][j]
    #         cells_new[row, col] = rep_point_counter
    #         rep_point_counter += 1
    # cells = cells_new
    # points = np.array(new_points)

    return points, cells, cell_adjacency, edges2cells