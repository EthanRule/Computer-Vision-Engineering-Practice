import numpy as np

image_patch = np.array([
    [100, 150, 200],
    [150, 200, 250],
    [200, 250, 255]
])

edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

def apply_kernel(image, kernel):
    return image_patch @ edge_kernel

print(apply_kernel(image_patch, edge_kernel))