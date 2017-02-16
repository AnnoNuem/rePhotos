import numpy as np
import cv2
import multiprocessing as mp
import ctypes

def morph_process(src_img, s_x_min, shared_dst, dst_shape, points_new, points_old, quads):
    """
    Multiprocessing process to morph image stripes.

    Args:
        src_img: Stripe of the original image which is to be morphed.
        s_x_min: Offset of image stripe. Used to determine where to add stripe
            in shared_dst.
        shared_dst: Shared destination image in which morphed image is saved.
        dst_shape: Size of the destination image.
        points_new: Corner locations of the quad grid in the destination image.
        points_old: Corner locations of the quad grid in the source image.
        quads: Indices of corners of the quads constituiting the mesh.
    """
    x_max = dst_shape[1] - 1 
    y_max = dst_shape[0] - 1

    dst_img = np.reshape(to_numpy_array(shared_dst), dst_shape)
    #print dst_img.shape
    for quad in quads:
        # clip quadpoints to img size
        c_p = lambda point: (min(max(point[0], 0,), x_max), \
                             min(max(point[1], 0), y_max))

        quad_old = [(points_old[quad[0]]), 
                    (points_old[quad[1]]), 
                    (points_old[quad[2]]), 
                    (points_old[quad[3]])]
        quad_new = [c_p(points_new[quad[0]]), 
                    c_p(points_new[quad[1]]), 
                    c_p(points_new[quad[2]]), 
                    c_p(points_new[quad[3]])]
        
        bbox = cv2.boundingRect(np.float32(quad_new))
        bbox_old = cv2.boundingRect(np.float32(quad_old))

        t1_rect = []
        t_rect = []
        for i in range(0, 4):
            t_rect.append(((quad_new[i][0] - bbox[0]), 
                           (quad_new[i][1] - bbox[1])))
            t1_rect.append(((quad_old[i][0] - bbox_old[0]),
                            (quad_old[i][1] - bbox_old[1])))

        # call to solver in getPerspectiveTransform is most expensive arround
        # ~1 order of magnitude compared to all other in loop called
        # functions
        warp_mat = cv2.getPerspectiveTransform(np.float32(t1_rect), 
                                               np.float32(t_rect))

         
        mask = np.zeros((bbox[3],bbox[2], 4), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(t_rect), (1,1,1), 4, 0)
        mask[:,:,3] = mask[:,:,2]
        tmp_img =  cv2.warpPerspective(src_img[quad_old[0][1] : quad_old[2][1], 
                   quad_old[0][0] - s_x_min : quad_old[2][0] - s_x_min], 
                   warp_mat, (bbox[2], bbox[3]), None, 
                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) 
        
        dst_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = \
           dst_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] *\
           (1-mask) + tmp_img * mask 


def to_numpy_array(mp_arr):
    """
    Create numpy array from multiprocessing array.
    """
    return np.frombuffer(mp_arr.get_obj())


def morph(src_img, points_old, points_new, quads, grid_size, processes=1):
    """
    Returns morphed image given points of old and new grid and quadindices. 
    Sourceimage is divided in stripes which are morphed parallel.

    Args:
        src_img: The image which will be morphed.
        points_old: Positions of grid points in src_img.
        points_new: Positions to where the old points are moved.
        quads: List of quad indices.
        grid_size: Distance between grid lines.
        processes: Number of multiprocessing.Processes which are spawend.

    Returns:
        img_morh: The morphed src_img.    
    """
    assert len(points_old) == len(points_new), "Point lists of different size."
    assert len(points_old) > 0, "Point lists are empty."
    
    # offset point for new image. New image is framed with grid size
    # to allow for quads ending outside of new image
    points_new = np.array(points_new + grid_size)

    # Allocate space for final output
    dst_shape = tuple(map(sum, zip(src_img.shape,\
                                   (2 * grid_size, 2 * grid_size, 0))))
    shared_dst = mp.Array(ctypes.c_double, np.prod(dst_shape))

    jobs = []
    chunk_size = quads.shape[0]/processes 
    for i in xrange(processes):
        # Send slices of src image to each process.
        # Assume that src mesh is a ordered rectangular mesh
        # Assume that quads are ordered in quad list.
        # Assume that points are ordered in quad
        x_min = (points_old[(quads[chunk_size *i])[0]])[0]
        x_max = (points_old[(quads[chunk_size *i + chunk_size - 1])[2]])[0]

        p = mp.Process(target=morph_process, args=(
            src_img[:,x_min:x_max,:], 
            x_min, 
            shared_dst, 
            dst_shape, 
            points_new, 
            points_old, 
            quads[chunk_size*i:chunk_size*i+chunk_size,:]))
         
        jobs.append(p)
        p.start()
    
    for j in jobs:
        j.join()

    img_morph = (np.reshape(to_numpy_array(shared_dst), dst_shape))\
                [grid_size:-grid_size, grid_size:-grid_size]
    
    return(img_morph)
