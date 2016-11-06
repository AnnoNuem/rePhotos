import numpy as np
import cv2
from image_helpers import statistic_canny
from image_helpers import adaptive_thresh
#from image_helpers import pst_wrapper
from image_helpers import line_intersect 

# Patch Size Divisor: Area arround line in which similar line is searched
psd = 60
scale_factor = 2

ADAPT_THRESH = 0
STAT_CANNY = 1
PST = 2

def get_line(p1, p2, img, method):
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
    if method == ADAPT_THRESH:
        patch_p  = adaptive_thresh(patch) 
    elif method == STAT_CANNY:
        patch_p  = statistic_canny(patch) 
    elif method == PST:
        patch_p  = pst_wrapper(patch) 

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
    min_line_length = np.sqrt((p_x_max - p_x_min) * (p_x_max - p_x_min) + (p_y_max - p_y_min) * (p_y_max - p_y_min)) * 0.5 
    max_gap = min_line_length/2
    cv2.imshow('linet', patch_p)
    lines = cv2.HoughLinesP(patch_p, 1, np.pi/180.0, 1, np.array([]), min_line_length, max_gap)

    #lines = cv2.HoughLines(patch_p, 1, np.pi/180.0, 0)

    #print lines.shape
    #cv2.imshow('sdf', np.uint8(lines*255))

    # return given points if no lines found, else:
    if lines is not None:

        #iterate over max ten best lines
        
        if True:
            # Generate weight image with values decreasing with distance from given line
            weight_img = np.ones(patch_p.shape, dtype=np.uint8)
            cv2.line(weight_img, (p1_o[0], p1_o[1]), (p2_o[0], p2_o[1]), 0, 1, 4)
            # Missing in python
            CV_DIST_MASK_PRECISE = 0 
            CV_DIST_L2 = 2
            weight_img = cv2.distanceTransform(weight_img, CV_DIST_L2, CV_DIST_MASK_PRECISE)
            w_min = weight_img.min()
            w_max = weight_img.max()
            weight_img = (1 - (weight_img - w_min)/(w_min - w_max)) * mask

            # Iterate over lines and find most similar line
            a,b,c = lines.shape
            line_img = np.empty(weight_img.shape, dtype=np.uint8)
            weights = np.empty((a), dtype=np.float32)
            tmp_img = np.zeros(weight_img.shape, dtype=np.uint8)
            for i in range(a):
                line_img[:] = 0
                cv2.line(line_img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 1, 1, 4)
                cv2.line(tmp_img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 255, 1, 4)
                # Multiply line_img with weight_img and sum all entries
                weights[i] = np.einsum('ij,ij->', line_img, weight_img)
        
        
        if False:
            a,b,c = lines.shape
            weights = np.empty((a), dtype=np.float32)
            tmp_img = np.zeros(weight_img.shape, dtype=np.uint8)
            for i in range(a):
                line_img = np.zeros(patch_p.shape, dtype=np.uint8)
                cv2.line(line_img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 255, 5, 4)
                cv2.line(line_img2, (lines[i][0][0], lines[i][0][1]), (lines[i][0][3], -lines[i][0][2]), 255, 5, 4)
                cv2.line(tmp_img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 255, 1, 4)
                #line_img = cv2.distanceTransform(line_img, CV_DIST_L2, CV_DIST_MASK_PRECISE)
                #line_img = line_img - line_img.min()
                #line_img = (1 - line_img/line_img.max()) * mask
                cv2.imshow('baf', line_img)
                cv2.imshow('baf2', line_img2)
                cv2.waitKey(0)
                # Multiply line_img with weight_img and sum all entries
                weights[i] = np.einsum('ij,ij->', line_img, patch_p)
        

        # Limit line length to length of user drawn line
        i_max_w = weights.argmax()
        p1_h = (lines[i_max_w][0][:2]).astype(np.float32) 
        p2_h = (lines[i_max_w][0][2:]).astype(np.float32)
        p1_o = p1_o.astype(np.float32)
        p2_o = p2_o.astype(np.float32)
        # Compute end points of orthagonal lines
        b1_o = p1_o + np.array([-(p1_o[1] - p2_o[1]), (p1_o[0] - p2_o[0])])
        b2_o = p2_o + np.array([-(p2_o[1] - p1_o[1]), (p2_o[0] - p1_o[0])])
        z1 = line_intersect(p1_o, b1_o, p1_h, p2_h)
        z2 = line_intersect(p2_o, b2_o, p2_h, p1_h)

        cv2.imshow('hough', tmp_img)

        p1 = (np.rint(z1 + offset)).astype(int)
        p2 = (np.rint(z2 + offset)).astype(int)

    
    return [p1[0], p1[1], p2[0], p2[1]]
