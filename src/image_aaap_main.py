import cv2 
import os, sys
import numpy as np
from scipy import sparse
from image_aaap import construct_mesh_energy
from image_aaap import build_regular_mesh
from image_aaap import sample_lines
from image_aaap import bilinear_point_in_quad_mesh
from image_aaap import deform_aaap 
from image_helpers import scale
from image_morphing import morph


location = os.path.abspath(os.path.dirname(sys.argv[0]))


def aaap_morph(src_img, dst_img, src_lines, dst_lines, grid_size=-1, 
        line_constraint_type=2, deform_energy_weights=np.array([1,0.0100, 0,0]),
        n_samples_per_grid=1):

    # scale images, create alpha channel for easy cropping
    print("Scaling...")
    src_img_alpha = np.ones((src_img.shape[0], src_img.shape[1], 4), np.float32) * 255
    src_img_alpha[:, :, 0:3] = np.float32(src_img[:,:,0:3])
    dst_img_alpha = np.ones((dst_img.shape[0], dst_img.shape[1], 4), np.float32) * 255
    dst_img_alpha[:, :, 0:3] = np.float32(dst_img[:,:,0:3])
    src_img_alpha, dst_img_alpha, src_lines, dst_lines = scale(src_img_alpha, dst_img_alpha, src_lines, dst_lines)

    # init grid 
    print("Init grid...")
    grid_points, quads, grid_shape = build_regular_mesh(src_img_alpha.shape[1], src_img_alpha.shape[0], grid_size)

    # create energy matrix
    print("Creating energy matrix...")
    L = construct_mesh_energy(grid_points, quads, deform_energy_weights)

    # discretisize lines
    print("Discretizesing lines...")
    src_points, dst_points = sample_lines(src_lines, dst_lines, float(n_samples_per_grid)/grid_size)

    # express points by quads
    print("Expressing points by quads...")
    Asrc = bilinear_point_in_quad_mesh(src_points, grid_points, quads, grid_shape)
    
    # deform grid
    print("Deforming grid...")
    y_p = deform_aaap(grid_points, Asrc, dst_points, L, line_constraint_type)
    y_p = y_p.T

    # morph image
    print("Morphing...")
    (src_img_morphed, dst_img_cropped, src_img_cropped) = morph(dst_img_alpha, src_img_alpha, grid_points, y_p, quads)


    # postprocess
    src_img_morphed = np.uint8(src_img_morphed[:, :, 0:3])
    dst_img_cropped = np.uint8(dst_img_cropped[:, :, 0:3])
    src_img_cropped = np.uint8(src_img_cropped[:, :, 0:3])


    return src_img_morphed, dst_img_cropped, src_img_cropped

