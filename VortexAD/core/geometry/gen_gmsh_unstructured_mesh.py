import numpy as np
import gmsh

def gen_gmsh_unstructured_mesh_quad(span_array, thickness_array, chord_array, name='mesh'):
    nc, ns = len(chord_array) - 1, len(span_array)
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.add(name)
    gmsh_kernel = gmsh.model.occ # geo or occ

    spanwise_steps = (span_array[-1] - span_array[0]) / (ns)
    print(spanwise_steps)

    # adding points
    point_tags = []
    for i in range(nc):
        gmsh_kernel.add_point(chord_array[i], span_array[0], thickness_array[i], meshSize=spanwise_steps, tag=i+1)
        point_tags.append(i+1)
    point_tags.append(point_tags[0])

    # translated_point_tags = []
    # for i in range(nc):
    #     asdf = gmsh_kernel.copy([(0, i+1)])
    #     gmsh_kernel.translate(asdf, 0, span_array[-1], 0)
    #     # extruded_line = gmsh_kernel.extrude([(1, line_tags[i])], 0., span_array[-1], 0.)
    #     translated_point_tags.append(asdf[0][1])
    # translated_point_tags.append(translated_point_tags[0])
    
    chordwise_line_tags = []
    chordwise_trans_line_tags = []
    spanwise_line_tags = []
    for i in range(nc):
        gmsh_kernel.add_line(point_tags[i], point_tags[i+1], tag=i+1)
        chordwise_line_tags.append(i+1)

        # gmsh_kernel.add_line(translated_point_tags[i], translated_point_tags[i+1], tag=i+1+nc)
        # chordwise_trans_line_tags.append(i+1)

    # for i in range(nc):
    #     gmsh_kernel.add_line(point_tags[i], translated_point_tags[i], tag=i+1+2*nc)
    #     spanwise_line_tags.append(i+1+2*nc)



    # for i in range(nc):

    # print(gmsh_kernel.getEntities(1))
    # num_nodes_circ = 15
    
    # gmsh_kernel.synchronize()

    for i in range(nc):
        asdf = gmsh_kernel.extrude([(1,chordwise_line_tags[i])], 0, (span_array[-1] - span_array[0]), 0)

    gmsh_kernel.synchronize()



    # for curve in chordwise_line_tags:
    #     print(curve)
    #     gmsh.model.mesh.setTransfiniteCurve(curve, 1)

    # exit()

    spanwise_interp = ns
    chordwise_interp = nc
    # for i in range(nc):
    #     gmsh.model.mesh.setTransfiniteCurve(spanwise_line_tags[i], spanwise_interp)

    # s1 = gmsh_kernel.addCurveLoop(chordwise_line_tags, 1)
    # s2 = gmsh_kernel.addCurveLoop(chordwise_trans_line_tags, 2)
    # s3 = gmsh_kernel.addCurveLoop(spanwise_line_tags, 3)
    # print(gmsh_kernel.getEntities(2))

    for surf in gmsh_kernel.getEntities(2):
        print(surf)
        gmsh.model.mesh.setTransfiniteSurface(surf[1])

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
    gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
    gmsh.option.setNumber('Mesh.ElementOrder', 1)
    # gmsh_kernel.removeAllDuplicates()
    gmsh_kernel.synchronize()
    gmsh.model.mesh.generate(2)


    gmsh.write(f'{name}.msh')
    gmsh.fltk.run()

    exit()

def gen_gmsh_unstructured_mesh(span_array, thickness_array, chord_array, name='mesh'):
    nc, ns = len(chord_array) - 1, len(span_array)
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.model.add(name)
    gmsh_kernel = gmsh.model.occ # geo or occ

    spanwise_steps = (span_array[-1] - span_array[0]) / (ns-1)

    # adding points
    point_tags = []
    for i in range(nc):
        gmsh_kernel.add_point(chord_array[i], span_array[0], thickness_array[i], meshSize=spanwise_steps, tag=i+1)
        point_tags.append(i+1)
    point_tags.append(point_tags[0])
    
    chordwise_line_tags = []
    for i in range(nc):
        gmsh_kernel.add_line(point_tags[i], point_tags[i+1], tag=i+1)
        chordwise_line_tags.append(i+1)
        
    gmsh_kernel.add_curve_loop(chordwise_line_tags, tag=1)
    gmsh_kernel.add_plane_surface([1], tag=1)

    gmsh_kernel.extrude([(2,1)], 0, (span_array[-1] - span_array[0]), 0)

    gmsh_kernel.synchronize()

    asdf = gmsh.model.getNormal(1, (0.5, 0.5))

    # gmsh.model.mesh.setOutwardOrientation(tag)
    gmsh.model.mesh.setReverse(2, 1, val=True) # reverses normal direction of surface 1 mesh

    gmsh.model.mesh.generate(2)

    gmsh.write(f'{name}.msh')
    gmsh.fltk.run()

    exit()


def convert_to_unstructured(mesh):
    nc, ns = mesh.shape[0], mesh.shape[1]
    nc_panels, ns_panels = nc-1, ns-1
    num_panels = nc_panels*ns_panels

    points = mesh.reshape((nc*ns,3))

    connectivity = np.zeros((num_panels, 4), dtype=int)
    panel_count = 0
    for i in range(nc_panels):
        for j in range(ns_panels):
            connectivity[panel_count,:] = np.array([int(i*ns+j), int(i*ns+j+1), int((i+1)*ns+j), int((i+1)*ns+j+1)])
            panel_count += 1

    return points, connectivity