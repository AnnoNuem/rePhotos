import cv2 
import matlab.engine
import os, sys
import numpy as np
from scipy import sparse
from image_aaap import construct_mesh_energy
from image_aaap import build_regular_mesh
from image_aaap import sample_lines
from image_aaap import bilinear_point_in_quad_mesh
from image_helpers import scale
from image_morphing import morph


location = os.path.abspath(os.path.dirname(sys.argv[0]))
dataname = 'data.mat'


def init_matlab():
    eng = matlab.engine.connect_matlab()
    eng.eval('clear',nargout=0)

    eng.workspace['mp_dataname'] = dataname
    eng.workspace['mp_location'] = location

    eng.eval('cd(mp_location);')
    eng.eval('load([mp_dataname]);', nargout=0)
    return eng


def aaap_morph(src_img, dst_img, src_lines, dst_lines, grid_size=50, 
        line_constraint_type=2, deform_energy_weights=np.array([1,0.0100, 0,0]),
        n_samples_per_grid=1):

    eng = init_matlab()

    # scale images, create alpha channel for easy cropping
    src_img_alpha = np.ones((src_img.shape[0], src_img.shape[1], 4), np.float32) * 255
    src_img_alpha[:, :, 0:3] = np.float32(src_img[:,:,0:3])
    dst_img_alpha = np.ones((dst_img.shape[0], dst_img.shape[1], 4), np.float32) * 255
    dst_img_alpha[:, :, 0:3] = np.float32(dst_img[:,:,0:3])
    src_img_alpha, dst_img_alpha, src_lines, dst_lines = scale(src_img_alpha, dst_img_alpha, src_lines, dst_lines)

    # init grid 
    grid_points, quads, grid_shape = build_regular_mesh(src_img_alpha.shape[1], src_img_alpha.shape[0], grid_size)
    eng.workspace['gridPoints'] = matlab.double(grid_points.tolist())
    eng.workspace['quads'] = matlab.double((quads + 1).tolist())
    eng.workspace['gridShape'] = grid_shape
    eng.workspace['gridSize'] = float(grid_size)

    # create energy matrix
    L = construct_mesh_energy(grid_points, quads, deform_energy_weights)

    # change coordinate sytem of lines to matlab images
    y_max = src_img_alpha.shape[0]
    for line in src_lines:
        line[1] = y_max - line[1]
        line[3] = y_max - line[3]
    for line in dst_lines:
        line[1] = y_max - line[1]
        line[3] = y_max - line[3]

    # discretisize lines
    src_points, dst_points = sample_lines(src_lines, dst_lines, float(n_samples_per_grid)/grid_size)

    # express points by quads
    Asrc = bilinear_point_in_quad_mesh(src_points, grid_points, quads, grid_shape)
    
    print Asrc

    # deform grid
    linesrc = matlab.double(src_lines)
    linedst = matlab.double(dst_lines)
    eng.workspace['linesrc'] = linedst
    eng.workspace['linedst'] = linesrc
    eng.workspace['lineConstraintType'] = line_constraint_type

    x, y, quads = eng.eval('test(gridPoints, quads, gridShape, linesrc, linedst, nSamplePerGrid, \
        lineConstraintType, deformEnergyWeights, gridSize)', nargout=3)

    # transform point coordinates from matlab to numpy
    points_old = []
    points_new = []
    max_x = dst_img_alpha.shape[1]
    max_y = dst_img_alpha.shape[0]
    for point in x:
        points_old.append((point[0], max_y - point[1]))
    for point in y:
        points_new.append((point[0], max_y - point[1]))

    # morph image
    (src_img_morphed, dst_img_cropped, src_img_cropped) = morph(dst_img_alpha, src_img_alpha, points_old, points_new, quads)

    # postprocess
    src_img_morphed = np.uint8(src_img_morphed[:, :, 0:3])
    dst_img_cropped = np.uint8(dst_img_cropped[:, :, 0:3])
    src_img_cropped = np.uint8(src_img_cropped[:, :, 0:3])

    eng.quit()
    return src_img_morphed, dst_img_cropped, src_img_cropped

