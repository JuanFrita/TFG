import numpy as np

def read_pts_old(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}")).astype(np.float32)

def read_pts(path):
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')
    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]
    """Convert entries from lists of strings to tuples of floats"""
    points = np.array([list([float(point) for point in coords]) for coords in coords_set]).astype(np.float32)
    return points

# Example usage
filename = 'D:/TFG/resources/Test_Training/test_pretrained/train/img_0.pts'
points = read_pts(filename)
print(points)