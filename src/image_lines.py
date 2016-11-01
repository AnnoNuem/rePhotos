import numpy as np
import cv2
from image_helpers import statistic_canny

# Patch Size Divisor: Area arround line in which similar line is searched
psd = 60

def get_line(p1, p2, img):
    """
    Search in image for most similar line of a given line.

    Args:
        p1: Start point of user drawn line.
        p2: End point of user drawn line.
        img: Image in which neares line in searched.

    Returns:
        [p1,p2]: List of two points of computed most similar line
    """
    
    # Get length and slope of line
    length = np.sqrt((p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[0] - p2[0]) * (p1[0] - p2[0])) 
    v = np.array([(p2[1] - p1[1])/length, (p2[0] - p1[0])/length])

    # Compute horizontal rectangle containing the patch arround the given line
    s = (img.shape[0] + img.shape[1])/ (2 * psd)
    m1 = p1 + s * (v * [ 1, -1] - v[::-1]) 
    m2 = p1 + s * (v * [ -1, 1] - v[::-1]) 
    m3 = p2 + s * (v * [ -1, 1] + v[::-1])
    m4 = p2 + s * (v * [ 1, -1] + v[::-1])
    p = np.array([m1, m2, m3, m4])
    x_min = int(max(0, p.T[1,:].min()))
    x_max = int(min(img.shape[0], p.T[1,:].max()))
    y_min = int(max(0, p.T[0,:].min()))
    y_max = int(min(img.shape[1], p.T[0,:].max()))

    patch = img[x_min: x_max, y_min: y_max]

    # Edge detection
    patch_p = statistic_canny(patch) 
    cv2.imshow('as2df', patch_p)

    # Generate mask of patch size
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=patch_p.dtype)
    cv2.fillPoly(mask, [p.astype(int)], (1))
    mask = mask[x_min: x_max, y_min: y_max]
    # Filter patch_p to only contain the patch area not the containing rectangle
    patch_p = patch_p * mask

    # Offset points
    offset = np.array([y_min, x_min])
    p1_o = p1 - offset
    p2_o = p2 - offset

    # Line detection
    p_x_min = min(p1_o[1],p2_o[1])
    p_x_max = max(p1_o[1],p2_o[1])
    p_y_min = min(p1_o[0],p2_o[0])
    p_y_max = max(p1_o[0],p2_o[0])
    min_line_length = np.sqrt((p_x_max - p_x_min) * (p_x_max - p_x_min) + (p_y_max - p_y_min) * (p_y_max - p_y_min))/1.8 
    max_gap = min_line_length/7
    lines = cv2.HoughLinesP(patch_p, 1, np.pi/180.0, 40, np.array([]), min_line_length, max_gap)

    # return given points if no lines found else:
    if lines is not None:
        # Generate weight image with values decreasing with distance from given line
        weight_img = np.ones(patch_p.shape, dtype=np.uint8)
        cv2.line(weight_img, (p1_o[0], p1_o[1]), (p2_o[0], p2_o[1]), 0, 1, 4)
        # Missing in python
        CV_DIST_MASK_PRECISE = 0 
        CV_DIST_L2 = 2
        weight_img = cv2.distanceTransform(weight_img, CV_DIST_L2, CV_DIST_MASK_PRECISE)
        weight_img = (1 - weight_img/weight_img.max()) * mask

        # Iterate over lines and find most similar line
        a,b,c = lines.shape
        line_img = np.empty(weight_img.shape, dtype=np.uint8)
        weights = np.empty((a), dtype=np.float32)
        for i in range(a):
            line_img[:] = 0
            cv2.line(line_img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 1, 1, 4)
            # Multiply line_img with weight_img and sum all entries
            weights[i] = np.einsum('ij,ij->', line_img, weight_img)
        
        p1 = lines[weights.argmax()][0][:2] + offset
        p2 = lines[weights.argmax()][0][2:] + offset
    
    return [p1[0], p1[1], p2[0], p2[1]]
