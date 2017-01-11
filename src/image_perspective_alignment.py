import numpy as np
import cv2
from image_helpers import weighted_average_point


def transform_lines(lines, transform_matrix):
    """
    Transforms a list of lines given a transformation matrix.

    Args:
        lines: the list of lines given as lines
        transform_matrix: the transformation matrix

    Returns:
        lines: transformed lines in float32
    """

    return cv2.perspectiveTransform(np.float32(lines).reshape(1,-2,2), 
        transform_matrix).reshape(-1, 4).tolist()


def transform_points(points, transform_matrix):
    """
    Transforms a list of points given a transformation matrix.
    Args:
        points: the list of points given as touples
        transform_matrix: the transformation matrix

    Returns:
        point_array: transformed points in int32
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
    """
    Aligns the two images with the best matching perspective transform given the 
    two point lists. Points and lines in their list are transformed as well.
    
    Args:
        img_1: Image 1
        img_2: Image 2
        points_img_1: marked points in image 1
        points_img_2: coresponding points in image 2
        alpha: 0 = align img_2 to img_1, 1 = align img_1 to img_2, 0.5 align 
               img_1 and img_2 to mean and points and lines acordingly. 
               If alpha is None the smaller image is transformed to bigger one.

    Returns:
        img_1: perspective transformed img_1    
        img_2: perspective transformed img_2    
        points_img_1: perspective transformed points_img_1
        points_img_2: perspective transformed points_img_2
        lines_img_1: perspective transformed lines_img_1
        lines_img_2: perspective transformed lines_img_2

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
    if alpha != 0:
        y, x, _ = img_1.shape
#       corners_img_1= [(0,0), (0,y), (x,y), (x,0)]
        transform_matrix_1, _ = cv2.findHomography(\
            np.vstack(points_img_1).astype(float), 
            np.vstack(dest_points).astype(float), 0)
        img_1 = cv2.warpPerspective(img_1, transform_matrix_1, 
            (x_max_dest, y_max_dest))
        points_img_1 = transform_points(points_img_1, transform_matrix_1)
        lines_img_1 = transform_lines(lines_img_1, transform_matrix_1)
#       corners_img_1 = transform_points(corners_img_1, transform_matrix_1)

    # Image 2
    if alpha != 1:
        y, x, _ = img_2.shape
#       corners_img_2= [(0,0), (0,y), (x,y), (x,0)]
        transform_matrix_2, _ = cv2.findHomography(\
            np.vstack(points_img_2).astype(float), 
            np.vstack(dest_points).astype(float), 0)
        img_2 = cv2.warpPerspective(img_2, transform_matrix_2, 
            (x_max_dest, y_max_dest))
        points_img_2 = transform_points(points_img_2, transform_matrix_2)
        lines_img_2 = transform_lines(lines_img_2, transform_matrix_2)
#       corners_img_2 = transform_points(corners_img_2, transform_matrix_2)

#    # Crop
#    x_min = max((corners_img_1[0])[0], (corners_img_1[1])[0], 
#                (corners_img_2[0])[0], (corners_img_2[1])[0])
#    y_min = max((corners_img_1[0])[1], (corners_img_1[3])[1], 
#                (corners_img_2[0])[1], (corners_img_2[3])[1])
#    x_max = min((corners_img_1[2])[0], (corners_img_1[3])[0], 
#                (corners_img_2[2])[0], (corners_img_2[3])[0])
#    y_max = min((corners_img_1[1])[1], (corners_img_1[2])[1], 
#                (corners_img_2[1])[1], (corners_img_2[2])[1])
#
#    points_img_1_cropped = []
#    for point in points_img_1:
#        x_cropped = point[0] - x_min
#        y_cropped = point[1] - y_min
#        if 0 <= x_cropped < x_max - x_min and 0 <= y_cropped < y_max - y_min:
#            points_img_1_cropped.append((x_cropped, y_cropped))
#
#    points_img_2_cropped = []
#    for point in points_img_2:
#        x_cropped = point[0] - x_min
#        y_cropped = point[1] - y_min
#        if 0 <= x_cropped < x_max - x_min and 0 <= y_cropped < y_max - y_min:
#            points_img_2_cropped.append((x_cropped, y_cropped))
#
#    img_1 = img_1[y_min:y_max, x_min:x_max, :]
#    img_2 = img_2[y_min:y_max, x_min:x_max, :]

    return img_1, img_2, points_img_1, points_img_2, lines_img_1, lines_img_2 
