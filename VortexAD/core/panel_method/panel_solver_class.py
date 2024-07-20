import numpy as np 
import csdl_alpha as csdl

from VortexAD.core.panel_method.unsteady_panel_solver import unsteady_panel_solver
from VortexAD.utils.cell_adjacency import find_cell_adjacency

class PanelSolver(object):
    def __init__(self, mode='source-doublet', BC='Dirichlet', unsteady=False, unstructured=False, dt=None, free_wake=False) -> None:
        self.mode = mode
        self.BC = BC        
        self.unsteady = unsteady
        self.mesh_mode = 'structured'
        if unstructured:
            self.mesh_mode = 'unstructured'
        self.surface_index = 0.

        self.dt = dt
        self.free_wake = free_wake

        self.surface_names = []
        self.points = []

        if unstructured:
            self.points = None
            self.cell_adjacency = None
            self.point_velocity = None
            self.TE_edges = None
        else:
            self.mesh = []
            self.mesh_velocity = []

        if unsteady and dt is None:
            raise TypeError('Must specify a time increment for unsteady solvers.')
        
        self._retrieve_output_names()
        
    def _retrieve_output_names(self):
        self.output_names = {
            'CL or L': 'per surface',
            'CDi or Di': 'per surface',
            'CP': '(nc, ns)',
            'panel forces': '(nc, ns)'

        }


    def add_structured_surface(self, mesh, mesh_velocity, name=None) -> None:
        if name is None:
            name = f'surface_{self.surface_index}'
            self.surface_names.append(name)
        self.surface_index += 1
        self.mesh.append(mesh)
        self.mesh_velocity.append(mesh_velocity)

    def add_unstructured_grid_data(self, points, cells, point_velocity) -> None:
        self.points = points
        self.point_velocity = point_velocity
        
        self.cell_adjacency = find_cell_adjacency(points, cells)

    def define_unstructured_surfaces(self, surfaces, TE_edges=None):
        self.surfaces = surfaces # pointer to elements of surfaces to get forces, etc.
        self.TE_edges = TE_edges # pointer to the TE edges to assign wake propagation & Kutta condition
        
    def set_outputs(self) -> None:
        pass

    def evaluate(self, dt, free_wake=False):
        # OPTION FOR 4 SOLVERS HERE:
        # steady or unsteady (prescribed/free)
        # structured or unstructured

        if self.unsteady:
            outputs = unsteady_panel_solver(
                (self.points, self.cell_adjacency),
                self.point_velocity,
                self.TE_edges,
                dt=dt,
                mesh_mode=self.mesh_mode,
                mode=self.mode,
                free_wake=free_wake,

            )
        else:
            raise NotImplementedError('Steady panel method has not been implemented yet.')




        return outputs