import numpy as np
import cv2
import os
import multiprocessing as mp
import ctypes

def morph_process(shared_src, shared_dst, src_shape, points_new, points_old, quads):

    print 'startet', os.getpid()

    x_max = src_shape[1] - 1 
    y_max = src_shape[0] - 1

    src_img = np.reshape(to_numpy_array(shared_src), src_shape)
    dst_img = np.reshape(to_numpy_array(shared_dst), src_shape)

    for quad in quads:
        # clip quadpoints to img size
        c_p = lambda point: (min(max(point[0], 0,), x_max), \
                             min(max(point[1], 0), y_max))

        quad_old = [c_p(points_old[int(quad[0])]), 
                    c_p(points_old[int(quad[1])]), 
                    c_p(points_old[int(quad[2])]), 
                    c_p(points_old[int(quad[3])])]
        quad_new = [c_p(points_new[int(quad[0])]), 
                    c_p(points_new[int(quad[1])]), 
                    c_p(points_new[int(quad[2])]), 
                    c_p(points_new[int(quad[3])])]
        
        bbox = cv2.boundingRect(np.float32(quad_new))
        bbox_old = cv2.boundingRect(np.float32(quad_old))

        t1_rect = []
        t_rect = []

        for i in range(0, 4):
            t_rect.append(((quad_new[i][0] - bbox[0]), (quad_new[i][1] - bbox[1])))
            t1_rect.append(((quad_old[i][0] - bbox_old[0]), (quad_old[i][1] - bbox_old[1])))
        warp_mat = cv2.getPerspectiveTransform(np.float32(t1_rect), 
                                               np.float32(t_rect))

         
        mask = np.zeros((bbox[3],bbox[2], 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(t_rect), (1,1,1), 4, 0)
        dst_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = \
            dst_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] * (1-mask) +\
            cv2.warpPerspective(src_img[quad_old[0][1] : quad_old[2][1], 
                                        quad_old[0][0] : quad_old[2][0]], 
                                        warp_mat, (bbox[2], bbox[3]), None, 
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_REPLICATE) * mask 

    #l.acquire()
    #img_morph += tmp_img
    #l.release()
    print 'ended', os.getpid()

def to_numpy_array(mp_arr):
    return np.frombuffer(mp_arr.get_obj())


def morph(src_img, points_old, points_new, quads, grid_size, scale=4, processes=4):
    """
    Returns morphed image given points of old and new grid and quads 
    """
    assert len(points_old) == len(points_new), "Point lists of different size."
    assert len(points_old) > 0, "Point lists are empty."

    # Scale src img for better results
    # Add frame of size grid_size arround image to accomodate for quads outside
    # image.
    if scale != 1:
        src_img = cv2.resize(src_img, (0,0), fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        points_old = np.array((points_old + grid_size ) * scale) 
        points_new = np.array((points_new + grid_size ) * scale)
        s_grid_size = scale * grid_size
    else:
        points_old = np.array(points_old + grid_size)
        points_new = np.array(points_new + grid_size)
        s_grid_size = grid_size

    # Allocate space for final output

    print 1
    s_src_img = np.zeros((src_img.shape[0] + 2 * s_grid_size, \
                          src_img.shape[1] + 2 * s_grid_size, \
                          src_img.shape[2]), dtype=np.float32)
    s_src_img[s_grid_size:-s_grid_size, s_grid_size:-s_grid_size] = src_img
    print 11

    shared_src = mp.Array(ctypes.c_double, s_src_img.size)
    shared_src_np = to_numpy_array(shared_src)
    shared_src_np[:] = s_src_img.flatten()[:]
    print 2

    shared_dst = mp.Array(ctypes.c_double, s_src_img.size)

    print 3



#    result_queue = Queue()
    #morphers = [morph_process(shared_src, s_src_img.shape, points_new, points_old, quad_block, x_max, y_max) for quad_block in np.vsplit(quads, processes)]
    #for mphr in morphers:
    #    p = Process(mphr)
    #    jobs.append(p)
    #    p.start()
    
#    jobs = [Process(mp) for mp in morphers]
#    for job in jobs: job.start()
#    for job in jobs: job.join()
#
#    for mp in morphers:
#        img_morph += result_queue.get()
#    results = [result_queue.get() for mp in morphers]
    jobs = []
   # l = Lock()
    chunk_size = quads.shape[0]/processes 
    #result_queue = Queue()
    #print chunk_size
    for i in xrange(processes):
        p = mp.Process(target=morph_process, args=(shared_src, shared_dst, s_src_img.shape, points_new, points_old, quads[chunk_size*i:chunk_size*i+chunk_size,:]))
        
        jobs.append(p)
        p.start()
    
    for j in jobs:
        j.join()
        
    img_morph = np.reshape(to_numpy_array(shared_dst), s_src_img.shape)
    img_morph = img_morph[s_grid_size:-s_grid_size, s_grid_size:-s_grid_size]
 #   img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, (5,5))
    
    if scale !=1:
        img_morph = cv2.resize(img_morph, (0,0), fx=1./scale, fy=1./scale, 
            interpolation=cv2.INTER_AREA)


    return(img_morph)
