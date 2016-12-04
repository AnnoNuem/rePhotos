import numpy as np
import cv2
    

def morph(src_img, points_old, points_new, quads, grid_size, scale=4):
    """
    Returns morphed image given points of old and new grid and quads 
    """
    assert len(points_old) == len(points_new), "Point lists have different size."
    assert len(points_old) > 0, "Point lists are empty."

    if scale != 1:
        src_img = cv2.resize(src_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        points_old = np.array((points_old + grid_size ) * scale) 
        points_new = np.array((points_new + grid_size ) * scale)
        s_grid_size = scale * grid_size
    else:
        points_old = np.array(points_old + grid_size)
        points_new = np.array(points_new + grid_size)
        s_grid_size = grid_size

    # Allocate space for final output
    img_morph = np.zeros((src_img.shape[0] + 2 * s_grid_size, src_img.shape[1] + 2 * s_grid_size, src_img.shape[2]), dtype=np.float32)
    s_src_img = np.zeros(img_morph.shape)
    s_src_img[s_grid_size:-s_grid_size, s_grid_size:-s_grid_size] = src_img

    x_max = img_morph.shape[1] - 1 
    y_max = img_morph.shape[0] - 1
    for quad in quads:
        # clip quadpoints to img size
        c_p = lambda point: (min(max(point[0], 0,), x_max), min(max(point[1], 0), y_max))

        quad_old = [c_p(points_old[int(quad[0])]), 
                    c_p(points_old[int(quad[1])]), 
                    c_p(points_old[int(quad[2])]), 
                    c_p(points_old[int(quad[3])])]
        quad_new = [c_p(points_new[int(quad[0])]), 
                    c_p(points_new[int(quad[1])]), 
                    c_p(points_new[int(quad[2])]), 
                    c_p(points_new[int(quad[3])])]
        #morph_quad(src_img, img_morph, quad_old, quad_new)
        
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
        #img_morph[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] += mask*255
#        cv2.imshow('asgf', img_morph)
#        cv2.waitKey(0)
        img_morph[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = \
            img_morph[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] * (1-mask) +\
            cv2.warpPerspective(s_src_img[quad_old[0][1] : quad_old[2][1], 
                                        quad_old[0][0] : quad_old[2][0]], 
                                        warp_mat, (bbox[2], bbox[3]), None, 
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_REFLECT) * mask 

#        img_morph[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] += \
#            cv2.warpPerspective(s_src_img[quad_old[0][1] : quad_old[2][1], 
#                                        quad_old[0][0] : quad_old[2][0]], 
#                                        warp_mat, (bbox[2], bbox[3]), None, 
#                                        flags=cv2.INTER_NEAREST, 
#                                        borderMode=cv2.BORDER_CONSTANT)  
        

   
    img_morph = img_morph[s_grid_size:-s_grid_size, s_grid_size:-s_grid_size]
 #   img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, (5,5))
    
    if scale !=1:
        img_morph = cv2.resize(img_morph, (0,0), fx=1./scale, fy=1./scale, 
            interpolation=cv2.INTER_AREA)


    return(img_morph)
