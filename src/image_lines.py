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

def lim_line_length(p1_h, p2_h, p1_o, p2_o):
    # Limit line length to length of user drawn line
    p1_o = np.array(p1_o, dtype=np.float32)
    p2_o = np.array(p2_o, dtype=np.float32)
    p1_h = np.array(p1_h, dtype=np.float32)
    p2_h = np.array(p2_h, dtype=np.float32)
    # Compute end points of orthagonal lines
    b1_o = p1_o + np.array([-(p1_o[1] - p2_o[1]), (p1_o[0] - p2_o[0])])
    b2_o = p2_o + np.array([-(p2_o[1] - p1_o[1]), (p2_o[0] - p1_o[0])])
    z1 = line_intersect(p1_o, b1_o, p1_h, p2_h)
    z2 = line_intersect(p2_o, b2_o, p2_h, p1_h)
    z1 = (np.rint(z1)).astype(int)
    z2 = (np.rint(z2)).astype(int)
    return z1, z2


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
    #lines = cv2.HoughLinesP(patch_p, 1, np.pi/180.0, 1, np.array([]), min_line_length, max_gap)
    lines = cv2.HoughLines(patch_p, 1, np.pi/180.0, 10, np.array([]), 0, 0)


    # return given points if no lines found, else:
    if lines is not None:
        ### hough standard
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

            #iterate over max ten best lines
            nl = lines.shape[0]
            nl = min(nl, 10)
            tmp_img = np.zeros(patch_p.shape, dtype=np.uint8)
            line_img = np.empty(patch_p.shape, dtype=np.uint8)
            weights_patch = np.empty((nl), dtype=np.float32)
            weights_user = np.empty((nl), dtype=np.float32)
            line_segs = np.empty((nl), dtype=object)
            i, j = 0, 0
            theta_o = np.abs(np.arctan(float(p2_o[0] - p1_o[0]) / float(p2_o[1] - p1_o[1])))
            print theta_o 
            while  j < nl and i < 10 :
                # allow only lines with roughly the same slope as user drawn lines
                rho = lines[j][0][0]
                theta = lines[j][0][1]
                j += 1

                print i
                print theta 
                if np.abs(theta_o - theta) < .2:
                    line_img[:] = 0
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0, y0 = a * rho, b * rho
                    # TODO change 1000 to pshape
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                    pt1, pt2 = lim_line_length(pt1, pt2, p1_o, p2_o)
                    cv2.line(tmp_img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, 1, 4)
                    cv2.line(line_img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, 1, 4)
                    weights_patch[i] = np.einsum('ij,ij->', line_img, patch_p)
                    weights_user[i] = np.einsum('ij,ij->', line_img, weight_img)
                    line_segs[i] = (pt1, pt2)
                    i += 1
            weights_user = (weights_user - weights_user.min()) / (weights_user.max() - weights_user.min())
            weights_patch= (weights_patch - weights_patch.min()) / (weights_patch.max() - weights_patch.min())
            print weights_user
            print weights_patch
            weights = weights_patch + 0.5 * weights_user
            print weights
            print
            i = weights.argmax()
            p1 = line_segs[i][0] + offset
            p2 = line_segs[i][1] + offset
        
        ###  hough probabilistic
        if False:
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
        


        cv2.imshow('hough', tmp_img)


    
    return [p1[0], p1[1], p2[0], p2[1]]
