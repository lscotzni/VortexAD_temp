import meshio

filename = 'naca0012_mesh.msh'
mesh = meshio.read(
    filename,  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)
# mesh.points, mesh.cells, mesh.cells_dict, ...

points = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict

dict_keys = list(cells_dict.keys())

print(f'points shape: {points.shape}')
print(f'cell length: {len(cells)}')
print(f'vertex length: {cells_dict[dict_keys[0]].shape}')
print(f'line length: {cells_dict[dict_keys[1]].shape}')
print(f'quad length: {cells_dict[dict_keys[2]].shape}')


# mesh.vtk.read() is also possible