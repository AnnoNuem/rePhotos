import numpy as np
import cv2
from image_helpers import get_corners 
from image_helpers import weighted_average_point
from image_helpers import get_crop_indices


def apply_affine_transform(src, src_quad, dst_quad, size):
    """
    Apply affine transform calculated using src_quad and dst_quad to src and output an image of size.
    """
    # Given a pair of quads, find the affine transform. 3 points give unique solution 
    warp_mat = cv2.getAffineTransform(np.float32(src_quad[0:3]), np.float32(dst_quad[0:3]))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_quad(img1, img, t1, t):
    """
    Warps and alpha blends quad regions from img1 and img2 to img.
    """
    # Find bounding rectangle for each quad
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t_rect = []

    for i in range(0, 4):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))

    # Get mask by filling quadangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r[2], r[3])
    img_rect = apply_affine_transform(img1_rect, t1_rect, t_rect, size)

    # Copy quadangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask

def morph(src_img, dst_img, points_old, points_new, quads):
    """
    Returns morphed image given points of old and new grid and quads 
    """
    assert len(points_old) == len(points_new), "Point lists have different size."
    assert len(points_old) > 0, "Point lists are empty."

    # Allocate space for final output
    img_morph = np.zeros((max(src_img.shape[0], dst_img.shape[0]), max(src_img.shape[1], dst_img.shape[1]),
        max(src_img.shape[2], dst_img.shape[2])), dtype=src_img.dtype)

    x_max = img_morph.shape[1]
    y_max = img_morph.shape[0]
    for quad in quads:
        # -1 to account for matlab indices begin with 1
        a = int(quad[0]) - 1
        b = int(quad[1]) - 1
        c = int(quad[2]) - 1
        d = int(quad[3]) - 1 
        if 0 <= points_old[a][0] < x_max and 0 <= points_old[a][1] < y_max and \
            0 <= points_old[b][0] < x_max and 0 <= points_old[b][1] < y_max and \
            0 <= points_old[c][0] < x_max and 0 <= points_old[c][1] < y_max and \
            0 <= points_old[d][0] < x_max and 0 <= points_old[d][1] < y_max and \
            0 <= points_new[a][0] < x_max and 0 <= points_new[a][1] < y_max and \
            0 <= points_new[b][0] < x_max and 0 <= points_new[b][1] < y_max and \
            0 <= points_new[c][0] < x_max and 0 <= points_new[c][1] < y_max and \
            0 <= points_new[d][0] < x_max and 0 <= points_new[d][1] < y_max:
            quad_old = [points_old[a], points_old[b], points_old[c], points_old[d]]
            quad_new = [points_new[a], points_new[b], points_new[c], points_new[d]]
            
            quad_old_zipped = list(zip(*quad_old))
            quad_old_min_x = int(min(quad_old_zipped[0])) 
            quad_old_max_x = int(max(quad_old_zipped[0]))
            quad_old_min_y = int(min(quad_old_zipped[1])) 
            quad_old_max_y = int(max(quad_old_zipped[1]))
            quad_img = src_img[quad_old_min_y:quad_old_max_y, 
                quad_old_min_x:quad_old_max_x]
            
            morph_quad(src_img, img_morph, quad_old, quad_new)

    # Crop images
    x_min_2, x_max_2, y_min_2, y_max_2 = get_crop_indices(dst_img)
    x_min_1, x_max_1, y_min_1, y_max_1 = get_crop_indices(img_morph)
    x_min, x_max, y_min, y_max = max(x_min_1, x_min_2), min(x_max_1, x_max_2), max(y_min_1, y_min_2), min(y_max_1, y_max_2)

    cv2.imshow('sdaf', np.uint8(img_morph-1))
    cv2.imshow('sdddaf', np.uint8(dst_img-1))
    #return images_cropped
    return (img_morph[y_min_1:y_max_1, x_min_1: x_max_1, :], 
        dst_img[y_min_1:y_max_1, x_min_1: x_max_1, :],
        src_img[y_min_1:y_max_1, x_min_1: x_max_1, :])
