import numpy as np
import cv2
from image_helpers import scale
from image_helpers import get_corners 
from image_helpers import weighted_average_point
from image_helpers import get_crop_indices


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


def morph(img1, img2, points_img1, points_img2, alpha=0.5, steps=2):
    """
    Returns list of morphed images.
    """
    assert 0 <= alpha <= 1, "Alpha not between 0 and 1."
    assert len(points_img1) == len(points_img2), "Point lists have different size."
    assert len(points_img1) > 0, "Point lists are empty."
    assert steps > 1, "Number of steps has to be at least two."

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Add small number so only frame added by scaling and morphing is zero and can be cropped easily later
    img1 += 0.00000001
    img2 += 0.00000001

    # Scale
    img1, img2, points_img1, points_img2 = scale(img1, img2, points_img1, points_img2)

    # Add the corner points and middle point of edges to the point lists
    get_corners(img1, img2, points_img1, points_img2)

    # Check that all points are in respective image
    for point in points_img1:
        assert 0 <= point[0] < img1.shape[1] and 0 <= point[1] < img1.shape[0], "Point %s outside image 1!" % (point,)
    for point in points_img2:
        assert 0 <= point[0] < img2.shape[1] and 0 <= point[1] < img2.shape[0], "Point %s outside image 2!" % (point,)

    # Compute weighted average point coordinates
    points = []
    for i in range(0, len(points_img1)):
        points.append(weighted_average_point(points_img1[i], points_img2[i], alpha))
    
    rect = (0, 0, max(img1.shape[1], img2.shape[1]), max(img1.shape[0], img2.shape[0]))
    indices_tri = get_indices(rect, points)

    # Morph
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
        # Add images to list
        images.append(np.copy(img_morph))
    
    # Crop images
    # Either first or last image needs max crop
    x_min_1, x_max_1, y_min_1, y_max_1 = get_crop_indices(images[0])
    x_min_2, x_max_2, y_min_2, y_max_2 = get_crop_indices(images[-1])
    x_min, x_max, y_min, y_max = max(x_min_1, x_min_2), min(x_max_1, x_max_2), max(y_min_1, y_min_2), min(y_max_1, y_max_2)
    images_cropped = []
    for image in images:
        #images_cropped.append(np.uint8(image - 0.00000001))
        images_cropped.append(np.uint8(image[y_min:y_max, x_min:x_max, : ] - 0.00000001))
    return images_cropped
