import numpy as np

def build_regular_mesh(width, height, grid_size):

    x = np.int64(np.arange(0, height + grid_size, grid_size))
    y = np.int64(np.arange(0, width + grid_size, grid_size))

    a,b = np.meshgrid(x,y)

    grid_points = np.zeros((a.size,2), dtype=a.dtype)
    grid_points[:,1] = a.reshape(a.size) 
    grid_points[:,0] = b.reshape(b.size)

    n = y.size
    m = x.size

    quads = np.zeros((grid_points.shape[0] - x.size - y.size + 1, 4), dtype=np.int64) 
    quads[:, 0] = np.reshape(np.tile(np.arange(0, m - 1, 1),[n-1, 1]) +  m* np.tile(np.arange(0, n - 1, 1),[m - 1, 1]).T, [-1, 1]).T
    quads[:, 1] = np.reshape(np.tile(np.arange(0, m - 1, 1),[n-1, 1]) +  m* np.tile(np.arange(1, n,     1),[m - 1, 1]).T, [-1, 1]).T
    quads[:, 2] = np.reshape(np.tile(np.arange(1, m,     1),[n-1, 1]) +  m* np.tile(np.arange(1, n,     1),[m - 1, 1]).T, [-1, 1]).T
    quads[:, 3] = np.reshape(np.tile(np.arange(1, m,     1),[n-1, 1]) +  m* np.tile(np.arange(0, n - 1, 1),[m - 1, 1]).T, [-1, 1]).T
    
    return grid_points, quads, (m,n)


