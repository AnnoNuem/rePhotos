import numpy as np
from scipy.sparse import csc_matrix

def build_regular_mesh(width, height, grid_size):
    """
    creates meshgrid of given width, height and distance between grid points.

    Returns:
        grid_points: Array of points of the grid
        quads: quads spanning the grid
        (m,n): dimension of the grid
    """

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


def construct_mesh_energy(grid_points, quads, deform_energy_weights):
    """
    Create quadratic enery matrix for aaap deformation of quad mesh.
    Returns:
        L: Sparse matrixs coresponding to aaap quadratic energy.
    """

    nv = grid_points.shape[0]
    n = quads.shape[0]
   
    Ais = np.empty(n * 16)
    Ajs = np.empty(n * 16)
    As = np.empty(n * 16, dtype=complex)

    A1 = np.array([[1, -1, 1, -1],
                  [-1, 1, -1, 1],
                  [1, -1, 1, -1],
                  [ -1, 1, -1, 1]], dtype=complex)/4
    A2 = np.array([[1, 1j, -1, -1],
                  [-1j, 1, 1j, -1],
                  [-1, -1j, 1, 1j],
                  [ 1j, -1, -1j, 1]])/4
    A = (A1 * deform_energy_weights[0] + A2 * deform_energy_weights[1]) * 2
    A = A.T.reshape(A.size) 

    for i in range(n):
        vvi = quads[i,:]
        nvv = vvi.size

        vvii = np.tile(vvi, (nvv, 1))

        Ais[i*16:i*16+16] = vvii.reshape(vvii.size)
        Ajs[i*16:i*16+16] = vvii.T.reshape(vvii.size)
        As[i*16:i*16+16] = A
    
    L = csc_matrix((As, (Ais, Ajs)), shape=(nv, nv))
    L.eliminate_zeros()
    return L


def sample_lines(src_lines, dst_lines, sample_rate):
    """
    Samples points from line pairs
    """
    assert len(src_lines) == len(dst_lines), "Different number of lines per image."
    assert len(src_lines) > 0, "No lines."

    #TODO switch tp src dst instead dst src but matlab implemenation is like this
    line_pairs = np.concatenate((np.array(dst_lines,dtype=np.double), np.array(src_lines)), axis=1)

    line_src = line_pairs[:, 0::2] + 1j* line_pairs[:, 1::2]

    fGenWt = lambda n: np.array((np.arange(n-1, -1, -1), np.arange(0, n))).T/(n-1)

    fNSampleAdaptive = lambda pq: max(2, np.ceil(sample_rate * abs(pq[0] -pq[1])))

    pts = [ np.dot(fGenWt(fNSampleAdaptive(line_src[i,0:2])), line_src[i,:].reshape((2,-1)).T) for i in range(0,line_src.shape[0])]
    p1 =  pts[0][:, 0]
    p2 = []
    p2.append(pts[0][:, 1])
    for i in range(1, len(pts)):
        p1 = np.concatenate((p1, pts[i][:, 0]))
        p2.append(pts[i][:, 1])
  
    return p1, p2


def bilinear_point_in_quad_mesh(pts, X, P, qmSize):
    """
    Express points in a quad mesh as the convex combination of there 
    containing quads, using bilinear weights
    A = bilinearPointInQuadMesh(pts, X, P, qmSize)
    Args: 
        pts: points that are to be expressed as bilinear combinations of 
        the quadmesh vertices
        X, P: the vertices/connectivy of the quadmesh
        qmSize: size (rows/columns of quads) of the quadmesh, that is 
        consctructed to cover some image plane
    Returns:
        A: a matrix that gives the weights for the points as combinations 
        of the quadmesh vertices, i.e. A*X = pts
    """ 
    
    if pts.dtype == 'complex128':
        pts = np.array([pts.real, pts.imag])
        
    nx = X.shape[0]
    npts = pts.shape[1]

    # suboptimal but python has no minmax function
    bbox = np.array((np.amin(X, axis=0), np.amax(X, axis=0)))
    peak_dist = np.ptp(X, axis=0)

    eps = np.finfo(float).eps
    qij = np.array((np.ceil((pts[0, :] + eps - bbox[0, 0]) * (qmSize[1] - 1) / peak_dist[0]), 
                    np.ceil((pts[1, :] + eps - bbox[0, 1]) * (qmSize[0] - 1) / peak_dist[1])))

    q = (((qij[0, :] - 1) * (qmSize[0] - 1) + qij[1, :]) - 1).astype(int)

    wx = (pts[0, :] - X[P[q, 0], 0]) / (X[P[q, 1], 0] - X[P[q, 0], 0])
    wy = (pts[1, :] - X[P[q, 0], 1]) / (X[P[q, 3], 1] - X[P[q, 0], 1])

    Ascr = csc_matrix((np.array(((1 - wx) * (1 - wy), wx * (1 - wy),  wx * wy, (1 - wx) * wy )).T.flatten(), 
        (np.tile((np.arange(0, npts , 1)), [4,1]).T.flatten(),  P[q, :].flatten())), shape=(npts, nx))
    Ascr.eliminate_zeros()
    return Ascr


def deform_aaap(x, Asrc, pdst, L, line_constraint_type):
    """
    AAAP/ASAP deform a quadmesh with line constraints
    Args: 
        x: geometry of the original quadmesh
        Asrc: matrix that express lines (sampled points on lines) as linear 
        combinations of x
        pdst: target positions of the lines (sampled points on them), each 
        cell element corresponds to one line
        L: AAAP/ASAP energy of the quadmesh
        flexLineConstraints: constraint type of each line
    Returns:
        y: geometry of the deformed quadmesh
    """

    if x.dtype != 'complex128':
        x = x[:, 0] + 1j * x[:, 1]

    nv =  x.shape[0]

    B1 = []
    nb = 0
    
    #print Asrc
    if line_constraint_type > 0:
        n_samples_in_line = np.zeros((len(pdst) + 1))
        n_samples_in_line[1:] = np.array([line.size for line in pdst])
        AIdxs = np.cumsum(n_samples_in_line)
        n_lines = len(pdst)

        for i in range(0, n_lines):
            a = (pdst[i])[0]
            b = (pdst[i])[-1]

            if line_constraint_type == 2:
    #            print AIdxs[i]
    #            print AIdxs[i+1] -1
                A1 = Asrc[AIdxs[i]:AIdxs[i+1] -1, :]
    #            print A1
    return  -1
