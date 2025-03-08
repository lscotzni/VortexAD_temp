import numpy as np

def find_cell_adjacency(points, cells):
    '''
    points is a numpy array of shape (num_points, 3) where the 3 is the x,y,z coordinates
    cells is a numpy array of shape (num_cells, 3) where the 3 is the node indices of the cell
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
        1

    # return edges2cells

    # Finding neighboring cells (cell index is the dict key)
    cell_adjacency = {i: [] for i in range(num_cells)}
    for edge in edges2cells.keys():
        cell_pairs = edges2cells[edge]
        if len(cell_pairs) < 2:
            continue # means this is an edge on the border of the mesh
        cell_adjacency[cell_pairs[0]].append(cell_pairs[1])
        cell_adjacency[cell_pairs[1]].append(cell_pairs[0])

    cell_adjacency = np.array(list(cell_adjacency.values()))

    # Finding elements corresponding to each node/vertex
    points2cells = {i: [] for i in range(num_pts)}
    for c, cell in enumerate(cells):
        for ind in cell:
            points2cells[ind].append(c)

    return points, cells, cell_adjacency, edges2cells, points2cells