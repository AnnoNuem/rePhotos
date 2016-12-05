import cv2 
import numpy as np
from scipy import sparse
from image_aaap import construct_mesh_energy
from image_aaap import build_regular_mesh
from image_aaap import sample_lines
from image_aaap import bilinear_point_in_quad_mesh
from image_aaap import deform_aaap 
from image_helpers import scale
from image_helpers import unsharp_mask
from image_helpers import get_crop_idx
from image_morphing import morph


def draw_frame(img, x_min, x_max, y_min, y_max):
    thickness = int((img.shape[0] + img.shape[1]) / 900  ) + 1
    lineType = 8
    color = (255,255,255)
    cv2.line(img, (x_min, y_min), (x_min, y_max), color, thickness, lineType )
    cv2.line(img, (x_min, y_max), (x_max, y_max), color, thickness, lineType )
    cv2.line(img, (x_max, y_max), (x_max, y_min), color, thickness, lineType )
    cv2.line(img, (x_max, y_min), (x_min, y_min), color, thickness, lineType )


def aaap_morph(src_img, dst_img, src_lines, dst_lines, grid_size=15, 
        line_constraint_type=2, deform_energy_weights=np.array([1,0.0100, 0,0]),
        n_samples_per_grid=1):

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

    Returns:
        src_img_morphed: Warped and cropped source image.
        dst_img_cropped: Destination image cropped to same size as src_img.
        src_img_cropped: Source image cropped to evaluate warp.
    """

    src_img = np.float32(src_img)
    dst_img = np.float32(dst_img)
    # scale images, create alpha channel for easy cropping
    print("Scaling...")
    src_img, dst_img, src_lines, dst_lines, x_max, y_max = \
        scale(src_img, dst_img, src_lines, dst_lines)
    #x_max = dst_img.shape[1]
    #y_max = dst_img.shape[0]


    print src_img.shape
    print dst_img.shape
    # init grid 
    print("Init grid...")
    grid_points, quads, grid_shape = build_regular_mesh(src_img.shape[1],
        src_img.shape[0], grid_size)

    print grid_points.shape
    # create energy matrix
    print("Creating energy matrix...")
    L = construct_mesh_energy(grid_points, quads, deform_energy_weights)

    # discretisize lines
    print("Discretizesing lines...")
    src_points, dst_points = sample_lines(src_lines, dst_lines, 
        float(n_samples_per_grid)/grid_size)

    # express points by quads
    print("Expressing points by quads...")
    Asrc = bilinear_point_in_quad_mesh(src_points, grid_points, quads, 
        grid_shape)
    
    # deform grid
    print("Deforming grid...")
    y_p = deform_aaap(grid_points, Asrc, dst_points, L, line_constraint_type).T


    # morph image
    print("Morphing...")
    src_img_morphed = morph(dst_img, grid_points, y_p, quads, grid_size, scale = 4)

    # Crop images
    print("Compute crop...")
    c_idx = get_crop_idx(y_p, grid_shape, src_img_morphed.shape, x_max, y_max) 

    print("Postprocess...")

    #(img_morph[y_min:y_max, x_min: x_max, :], 
    #    dst_img[y_min:y_max, x_min: x_max, :],
    #    src_img[y_min:y_max, x_min: x_max, :])


    src_img_morphed = unsharp_mask(src_img_morphed, 1, .7)

    draw_frame(src_img_morphed, c_idx[0], c_idx[2], c_idx[1], c_idx[3])
    draw_frame(dst_img, c_idx[0], c_idx[2], c_idx[1], c_idx[3])
    draw_frame(src_img, c_idx[0], c_idx[2], c_idx[1], c_idx[3])
    return np.uint8(src_img_morphed), np.uint8(src_img), np.uint8(dst_img)

