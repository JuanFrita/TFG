import numpy as np

def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}")).astype(np.float32)

# Example usage
filename = 'D:/TFG/src/resources/pts_examples/example.pts'
points = read_pts(filename)
print(points)