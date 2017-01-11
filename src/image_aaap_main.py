import time
import cv2 
import numpy as np
import image_helpers as i_h
import image_aaap as i_aaap 
from scipy import sparse
from image_morphing import morph


def aaap_morph(src_img, dst_img, src_lines, dst_lines, grid_size=15, 
        line_constraint_type=2, deform_energy_weights=np.array([1,0.0100, 0,0]),
        n_samples_per_grid=1, scale_factor=4):

    """
    Wrapper for As-Affine-As-Possible Warping.
    As described in 'Generalized As-Similar-As-Possible Warping with
    Applications in Digital Photography' by Chen and Gotsman.

    Args:
        src_img: Source image which will be warped to match destination image.
        dst_img: Destination image which is only scaled.
        src_lines: User drawn lines in the source image.
        dst_lines: User drawn lines in the destination image.
        grid_size: Distance between grid lines in pixels.
        line_constraint_type: 
            0: Fixed discretisation of lines.
            1: Flexible discretisations, points can move on line but endpoints
                are fixed.
            2: Flexible discretisation, all points including endpoints can move
                on line.
        deform_energy_weights: Weighting affinity of warping. 
            [alpha, beta, 0,0]. See paper for details.
        n_samples_per_grid = Number of line discretisation points per grid block.
        scale_factor: Scaling factor for first image and both line lists. 
                      Second image is not scaled since not used for computation.

    Returns:
        src_img_morphed: Warped and cropped source image.
        dst_img_cropped: Destination image cropped to same size as src_img.
        src_img_cropped: Source image cropped to evaluate warp.
    """

    grid_size = grid_size * scale_factor

    # init grid 
    i_h.vprint("Init grid...")
    grid_points, quads, grid_shape = i_aaap.build_regular_mesh(dst_img.shape[1],
        dst_img.shape[0], grid_size)

    # create energy matrix
    i_h.vprint("Creating energy matrix...")
    L = i_aaap.construct_mesh_energy(grid_points, quads, deform_energy_weights)

    # discretisize lines
    i_h.vprint("Discretizesing lines...")
    src_points, dst_points = i_aaap.sample_lines(dst_lines, src_lines, 
        float(n_samples_per_grid)/grid_size)

    # express points by quads
    i_h.vprint("Expressing points by quads...")
    Asrc = i_aaap.bilinear_point_in_quad_mesh(src_points, grid_points, quads, 
        grid_shape)
    
    # deform grid
    i_h.vprint("Deforming grid...")
    y_p = i_aaap.deform_aaap(grid_points, Asrc, dst_points, L, line_constraint_type).T


    # morph image
    i_h.vprint("Morphing...")
    t = time.time()
    src_img_morphed = morph(src_img, grid_points, y_p, quads, grid_size, 
                            processes=4)
    
    # Downscale images
    if scale_factor !=1:
        src_img_morphed = cv2.resize(src_img_morphed, (0,0), fx=1./scale_factor, 
                          fy=1./scale_factor, interpolation=cv2.INTER_AREA)
        src_img_cropped= cv2.resize(src_img, (0,0), fx=1./scale_factor, 
                 fy=1./scale_factor, interpolation=cv2.INTER_AREA)
        dst_img_cropped = cv2.resize(dst_img, (0,0), fx=1./scale_factor, 
                          fy=1./scale_factor, interpolation=cv2.INTER_AREA)
    else:
        src_img_cropped = src_img
        dst_img_cropped = dst_img

    # Crop images
    i_h.vprint("Compute crop...")
    #c_idx = i_h.get_crop_idx(y_p, grid_shape, src_img_morphed.shape, x_max, y_max) 

    c_idx = i_h.get_crop_idx(src_img_morphed[:,:,3] + dst_img_cropped[:,:,3])

    i_h.vprint("Postprocess...")

    cv2.namedWindow('src', cv2.WINDOW_NORMAL)
    cv2.imshow('src', src_img_morphed[:,:,3])
    cv2.resizeWindow('src', 640, 480)

    cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    cv2.imshow('dst', dst_img_cropped[:,:,3])
    cv2.resizeWindow('dst', 640, 480)

    cv2.waitKey()
    cv2.destroyAllWindows()
    src_img_morphed = src_img_morphed[:,:,0:-1]
    src_img_cropped = src_img_cropped[:,:,0:-1]
    dst_img_cropped = dst_img_cropped[:,:,0:-1]

    # sharpen image
    src_img_morphed = i_h.unsharp_mask(src_img_morphed, 1, .7)

#    return (np.uint8(src_img_morphed[c_idx[1]:c_idx[3],c_idx[0]:c_idx[2]]),
#            np.uint8(src_img[c_idx[1]:c_idx[3],c_idx[0]:c_idx[2]]),
#            np.uint8(dst_img[c_idx[1]:c_idx[3],c_idx[0]:c_idx[2]]))

    i_h.draw_rectangle(src_img_morphed, (c_idx[0], c_idx[1]), (c_idx[2], c_idx[3]))
    i_h.draw_rectangle(dst_img, (c_idx[0], c_idx[1]), (c_idx[2], c_idx[3]))
    i_h.draw_rectangle(src_img, (c_idx[0], c_idx[1]), (c_idx[2], c_idx[3]))

    return np.uint8(src_img_morphed), np.uint8(src_img_cropped),\
           np.uint8(dst_img_cropped)

