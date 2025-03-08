import numpy as np
import csdl_alpha as csdl

class VertexNormals(csdl.CustomExplicitOperation):

    def __init__(self, adjacent_panels):
        """Paraboloid function implemented as a custom explicit operation."""
        super().__init__()

        # assign parameters to the class
        self.adjacent_panels = adjacent_panels

    def evaluate(self, vertices, panel_normal):
        # assign method inputs to input dictionary
        self.declare_input('vertices', vertices)
        self.declare_input('panel_normal', panel_normal)

        # declare output variables
        vertex_normal = self.create_output('vertex_normal', vertices.shape)

        # declare any derivative parameters
        self.declare_derivative_parameters('vertex_normal', 'vertices')
        self.declare_derivative_parameters('vertex_normal', 'panel_normal')

        return vertex_normal
    
    def compute(self, input_vals, output_vals):
        vertices = input_vals['vertices']
        panel_normal = input_vals['panel_normal']

        points2cells = self.adjacent_panels
        # dictionary where each key is a point and val is a list of cells
        vertex_normal = np.zeros(shape=vertices.shape)
        num_vertices = vertices.shape[1]

        # loop over vertices here
        for i in range(num_vertices):
            panels = points2cells[i]
            num_panels = len(panels)
            vec = np.sum(panel_normal[:,panels,:], axis=1)/num_panels
            vertex_normal[:,i,:] = vec

        output_vals['vertex_normal'] = vertex_normal

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        # NOTE: FIX DERIVATIVES
        pass
        # x = input_vals['x']
        # y = input_vals['y']
        # z = input_vals['z']

        # derivatives['f', 'x'] = 2*x - self.a + y
        # derivatives['f', 'y'] = 2*y + x + self.b

        # if self.return_g:
        #     derivatives['g', 'x'] = z*derivatives['f', 'x']
        #     derivatives['g', 'y'] = z*derivatives['f', 'x']
        #     derivatives['g', 'z'] = outputs_vals['f']