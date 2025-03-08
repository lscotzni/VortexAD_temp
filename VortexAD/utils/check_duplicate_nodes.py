import numpy as np
import warnings

def check_duplicate_nodes(points):
    '''
    points is a numpy array of shape (num_points, 3) where the 3 is the x,y,z coordinates
    '''
    dup_indices = []
    num_pts = points.shape[0]
    for i in range(num_pts-1):
        print(i)
        p0 = points[i,:]
        for j in range(i+1,num_pts):
            p1 = points[j,:]
            diff = np.linalg.norm(p1-p0)
            if diff < 1.e-10:
                dup_indices.append([i,j])

    if dup_indices: # not an empty list
        warnings.warn("Duplicate nodes may exist")

    return dup_indices
