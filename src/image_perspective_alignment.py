#!/usr/bin/env python
"""Functions to transform images, list of lines and list of points to match 
another image with a perspective transform matrix. The later is computed from 
two lists of points, which should be matched.
"""
import numpy as np
import cv2
from image_helpers import weighted_average_point


def transform_lines(lines, transform_matrix):
    """Transforms a list of lines given a transformation matrix.

    Parameters
    ----------
    lines : list
        The list of lines given as lists.
    transform_matrix : ndarray
        The transformation matrix.

    Returns
    -------
    lines : list
        Transformed lines in float32

    """
    if len(lines) > 0:
        return cv2.perspectiveTransform(np.float32(lines).reshape(1,-2,2), 
               transform_matrix).reshape(-1, 4).tolist()
    else:
        return []


def transform_points(points, transform_matrix):
    """Transforms a list of points given a transformation matrix.

    Parameters
    ----------
    points : list
        the list of points given as touples
    transform_matrix : ndarray
        the transformation matrix

    Returns
    -------
    point_array : list
        transformed points in int32

    """

    point_array = np.int32(cv2.perspectiveTransform(\
        np.float32(points).reshape(1,-1,2),
        transform_matrix).reshape(-1, 2)).tolist()
    
    return  [tuple(point) for point in point_array]


def perspective_align(img_1, 
                      img_2, 
                      points_img_1, 
                      points_img_2, 
                      lines_img_1, 
                      lines_img_2, 
                      alpha = None):
    """Aligns the two images with the best matching perspective transform given
    the two point lists. Points and lines in their list are transformed as well.

    Parameters
    ----------
    img_1 : ndarray
        Image 1
    img_2 : ndarray
        Image 2
    points_img_1 : list
        marked points in image 1
    points_img_2 : list
        coresponding points in image 2
    lines_img_1 : list
        marked lines in image 1    
    lines_img_2 : list
        coresponding lins in image 2
    alpha : float
        0 = align img_2 to img_1, 1 = align img_1 to img_2, 0.5 align
        img_1 and img_2 to mean and points and lines acordingly.
        If alpha is None the smaller image is transformed to bigger one. 
        (Default value = None)
        
    Returns
    -------
    img_1 : ndarray
        perspective transformed img_1
    img_2 : ndarray
        perspective transformed img_2
    points_img_1 : list
        perspective transformed points_img_1
    points_img_2 : list
        perspective transformed points_img_2
    lines_img_1 : list
        perspective transformed lines_img_1
    lines_img_2 : list
        perspective transformed lines_img_2

    """

    assert len(points_img_1) == len(points_img_2), "Pointlists of unequal length"
    assert len(points_img_1) > 3, "Not enough points to find homography"

    if (alpha == None):
        if (img_1.shape[0] * img_1.shape[1]) < (img_2.shape[0] * img_2.shape[1]):
            alpha = 1
        else:
            alpha = 0

    assert 0 <= alpha <= 1, "Alpha not between 0 and 1."

    x_max_dest = max(img_1.shape[1], img_2.shape[1])
    y_max_dest = max(img_1.shape[0], img_2.shape[0])

    # Find points of dest image
    if alpha == 0:
        dest_points = points_img_1
    elif alpha == 1:
        dest_points = points_img_2
    else:
        dest_points = []
        i = 0
        for point_1 in points_img_1:
            dest_points.append(weighted_average_point(point_1, points_img_2[i], 
                alpha))
            i += 1
        
    # Image 1
    transform_matrix_1 = None
    if alpha != 0:
        transform_matrix_1 = cv2.getPerspectiveTransform(\
                             np.vstack(points_img_1).astype(np.float32), 
                             np.vstack(dest_points).astype(np.float32))
        img_1 = cv2.warpPerspective(img_1, transform_matrix_1, 
            (x_max_dest, y_max_dest))
        points_img_1 = transform_points(points_img_1, transform_matrix_1)
        lines_img_1 = transform_lines(lines_img_1, transform_matrix_1)

    # Image 2
    transform_matrix_2 = None
    if alpha != 1:
        transform_matrix_2 = cv2.getPerspectiveTransform(\
                             np.vstack(points_img_2).astype(np.float32), 
                             np.vstack(dest_points).astype(np.float32))
        img_2 = cv2.warpPerspective(img_2, transform_matrix_2, 
            (x_max_dest, y_max_dest))
        points_img_2 = transform_points(points_img_2, transform_matrix_2)
        lines_img_2 = transform_lines(lines_img_2, transform_matrix_2)

    return img_1, img_2, points_img_1, points_img_2, lines_img_1, lines_img_2,\
           transform_matrix_1, transform_matrix_2
