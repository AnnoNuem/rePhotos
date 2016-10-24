import numpy as np

def build_regular_mesh(width, height, grid_size):
    m = np.ceil(height / grid_size) + 1
    n = np.ceil(width / grid_size) + 1

    x = np.int64(np.linspace(0, width, n-1))
    y = np.int64(np.linspace(0, height, m-1))

    a,b = np.meshgrid(x,y)

    
    print a.shape
    print a
    print b.shape
    print b



