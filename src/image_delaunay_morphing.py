import numpy as np
import cv2
from image_helpers import scale
from image_helpers import get_corners 
from image_helpers import weighted_average_point
from image_helpers import crop

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Apply affine transform calculated using src_tri and dst_tri to src and output an image of size.
    """
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img.
    """
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask


def get_indices(rect, points):
    """
    Returns indices of delaunay triangles.
    """
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
    triangle_list = subdiv.getTriangleList()

    indices_tri = []
    ind = [None] * 3
    pt = [None] * 3
    for triangle in triangle_list:
        pt[0] = (int(triangle[0])), int(triangle[1])
        pt[1] = (int(triangle[2])), int(triangle[3])
        pt[2] = (int(triangle[4])), int(triangle[5])
        if rect[0] <= (pt[0])[0] <= rect[2] and rect[1] <= (pt[0])[1] <= rect[3] and rect[0] <= (pt[1])[0] <= rect[2] \
                and rect[1] <= (pt[1])[1] <= rect[3] and rect[0] <= (pt[2])[0] <= rect[2] and rect[1] <= (pt[2])[1] <= rect[3]:
            for i in range(0, 3):
                for j in range(0, len(points)):
                    if pt[i][0] == points[j][0] and pt[i][1] == points[j][1]:
                        ind[i] = j
            indices_tri.append(list(ind))
    return indices_tri


def check_cropping(points, points_img1, points_img2, x_crop, y_crop, global_x_max, global_y_max, alpha):
    """
    Checks if zeros added by scaling are cropped in result images.
    Searches nearest point to padding in point list of image in which padding 
    with zeros was necessary for scaling. Does it independently for x 
    direction and y direction. Assumes padding start (*_crop) is moved as far
    as nearest point. Then checks if moved padding point is smaller than
    current crop mark (global_*_max).
    """
    # x
    # img1 is bigger
    if x_crop > 0:
        p1, i = min(((val, idx) for (idx, val) in enumerate(points_img2)),
                                 key=lambda p: abs(x_crop - (p[0])[0]))
        delta = p1[0] - (points[i])[0]
        global_x_max = x_crop - delta if x_crop - delta < global_x_max else global_x_max
    # img2 is bigger
    else:
        x_crop = abs(x_crop)
        p1, i = min(((val, idx) for (idx, val) in enumerate(points_img1)),
                                 key=lambda p: abs(x_crop - (p[0])[0]))
        delta = p1[0] - (points[i])[0]
        global_x_max = x_crop - delta if x_crop - delta < global_x_max else global_x_max

    # y
    # img1 is bigger
    if y_crop > 0:
        p1, i = min(((val, idx) for (idx, val) in enumerate(points_img2)),
                                 key=lambda p: abs(y_crop - (p[0])[1]))
        delta = p1[1] - (points[i])[1]
        global_y_max = y_crop - delta if y_crop - delta < global_y_max else global_y_max
    # img2 is bigger
    else:
        y_crop = abs(y_crop)
        p1, i = min(((val, idx) for (idx, val) in enumerate(points_img1)),
                                 key=lambda p: abs(y_crop - (p[0])[1]))
        delta = p1[1] - (points[i])[1]
        global_y_max = y_crop - delta if y_crop - delta < global_y_max else global_y_max

    return global_x_max, global_y_max


def morph(img1, img2, points_img1, points_img2, alpha=0.5, steps=2):
    """Returns list of morphed images."""

    assert 0 <= alpha <= 1, "Alpha not between 0 and 1."
    assert len(points_img1) == len(points_img2), "Point lists have different size."
    assert len(points_img1) > 0, "Point lists are empty."

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    img1, img2, points_img1, points_img2, x_crop, y_crop = scale(img1, img2, points_img1, points_img2)

    # Add the corner points and middle point of edges to the point lists
    global_x_min, global_x_max, global_y_min, global_y_max = get_corners(img1, img2, points_img1, points_img2, x_crop, y_crop, alpha)

    for point in points_img1:
        assert 0 <= point[0] < img1.shape[1] and 0 <= point[1] < img1.shape[0], "Point %s outside image 1!" % (point,)
    for point in points_img2:
        assert 0 <= point[0] < img2.shape[1] and 0 <= point[1] < img2.shape[0], "Point %s outside image 2!" % (point,)

    # Compute weighted average point coordinates
    points = []
    for i in range(0, len(points_img1)):
        points.append(weighted_average_point(points_img1[i], points_img2[i], alpha))
    
    # Check that zeros from scaling get cropped in result
    global_x_max, global_y_max = check_cropping(points, points_img1, points_img2, x_crop, y_crop, global_x_max, global_y_max, alpha)

    rect = (0, 0, max(img1.shape[1], img2.shape[1]), max(img1.shape[0], img2.shape[0]))
    indices_tri = get_indices(rect, points)

    images = []
    for a in np.linspace(0.0, 1.0, num=steps):
        # Allocate space for final output
        img_morph = np.zeros((max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1]), max(img1.shape[2], img2.shape[2])), dtype=img1.dtype)

        for ind in indices_tri:
            x = ind[0]
            y = ind[1]
            z = ind[2]

            t1 = [points_img1[x], points_img1[y], points_img1[z]]
            t2 = [points_img2[x], points_img2[y], points_img2[z]]
            t = [points[x], points[y], points[z]]

            # Morph one triangle at a time.
            morph_triangle(img1, img2, img_morph, t1, t2, t, a)
        # add cropped images to list
        images.append(np.copy(np.uint8(img_morph[int(global_y_min):int(global_y_max), int(global_x_min):int(global_x_max), :])))
    return images
