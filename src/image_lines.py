import numpy as np
from numpy import linalg as la
import cv2
from image_helpers import statistic_canny
from image_helpers import adaptive_thresh
from image_helpers import line_intersect 
from image_helpers import lce
from image_helpers import draw_line


def lim_line_length(p1_h, p2_h, p1_o, p2_o):
    """
    Limits length of line h to linesegment o.
    Computes a line segment of line h found by hough transform to the
    length of user drawn line segment o by computing the normals at start
    and end point of user drawn line segment and their intersection with
    hough line.
    
    Args:
        p1_h: First point on hough line.
        p2_h: Second point on hough line.
        p1_o: Startpoint of user drawn line segment.
        p2_o: Endpoint of user drawn line segment.

    Returns:
        z1: Startpoint of line segment of hough line.
        z2: Endpoint of line segment of hough line.
    """
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


def get_theta(p1, p2):
    """
    Computes gradient angle of line.

    Args:
        p1_o: First point of line.
        p2_o: Second point of line.

    Returns:
        theta: Gradient angle.
    """
    if p2[0] - p1[0] == 0:
        return 0
    else:
        m = (float(p2[1] - p1[1]) / float(p2[0] - p1[0]))
        return np.arctan(m) + np.pi/2


def weight_lines(patch_p, lines, p1_o, p2_o, number_of_lines=10, return_best_line=True):
    """
    Weights hough lines by similarity to edges.
    Generates line segments with length according to user drawn lines and 
    compares segments with edges in edge image. Select best matching line.
    Only lines with similar orientation to user drawn line are regarded.
    Only best ten lines fullfilling above criterium are considered.

    Args:
        patch_p: Edgeimage.
        lines: List of lines.
        p1_o: Startpoint of user drawn line.
        p2_o: Endpoint of user drawn line.

    Returns:
        p1: Startpoint of best matching line.
        p2: Endpoint of best matching line.
    """
    #iterate over max ten best lines
    nl = min(number_of_lines, lines.shape[0])
    line_img = np.empty(patch_p.shape, dtype=np.uint8)
    weights = np.empty((nl), dtype=np.float32)
    line_segs = np.empty((nl), dtype=object)
    i, j = 0, 0
    theta_o = get_theta(p1_o, p2_o)    

    while  j < lines.shape[0] and i < number_of_lines :
        rho = lines[j][0][0]
        theta = lines[j][0][1]
        j += 1
        # allow only lines with roughly the same slope as user drawn lines
        if np.abs(theta_o - theta) < .05 or \
            np.pi - np.abs(theta_o - theta) < 0.05:
            line_img[:] = 0
            a = np.cos(theta)
            b = np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            pt1, pt2 = lim_line_length(pt1, pt2, p1_o, p2_o)
            cv2.line(line_img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 255, 1, 4)
            weights[i] = np.einsum('ij,ij->', line_img, patch_p, dtype=np.float32)
            line_segs[i] = (pt1, pt2)
            i += 1

    if return_best_line:
        if i > 0:
            k = weights[:i].argmax()
            return line_segs[k][0], line_segs[k][1], weights[k]
        else:
            return p1_o, p2_o, 0
    else:
        
        return line_segs, weights


def get_patch(img, p1, p2, psd=70):
    # Get length and slope of line
    length = np.sqrt((p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[0] - p2[0]) * 
        (p1[0] - p2[0])) 
    v = np.array([(p2[1] - p1[1])/length, (p2[0] - p1[0])/length])

    # Compute rectangle with line in the middle and same rotation as line
    # and bounding box arround it
    s = (img.shape[0] + img.shape[1])/ (2 * psd)
    m1 = p1 + s * (v * [ 1, -1] - v[::-1]) 
    m2 = p1 + s * (v * [ -1, 1] - v[::-1]) 
    m3 = p2 + s * (v * [ -1, 1] + v[::-1])
    m4 = p2 + s * (v * [ 1, -1] + v[::-1])
    p = np.array([m1, m2, m3, m4], dtype=np.uint32)
    x_min = int(max(0, p.T[1,:].min()))
    x_max = int(min(img.shape[0], p.T[1,:].max()))
    y_min = int(max(0, p.T[0,:].min()))
    y_max = int(min(img.shape[1], p.T[0,:].max()))

    patch = img[x_min: x_max, y_min: y_max]

    # Generate mask of patch size
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=patch.dtype)
    cv2.fillPoly(mask, [p.astype(int)], (1))
    mask = mask[x_min: x_max, y_min: y_max]
    
    offset = np.array([y_min, x_min])

    return patch, mask, offset


def line_detect(img, mask=None, sigma=0.33, magic_n=100):
#    img = cv2.equalizeHist(img)

    # Edge dedection
    m = np.median(img)
    lower_bound = int(max(0, (1.0 - sigma) * m)) + magic_n
    upper_bound = int(min(255, (1.0 + sigma) * m)) + magic_n
    img = cv2.Canny(img, lower_bound, upper_bound, L2gradient=True)

    # Filter patch_p to only contain the patch area not the containing rectangle
    if mask is not None:
        img = img * np.uint8(mask)
    
    # Line detection
    return cv2.HoughLines(img, 1, np.pi/180.0, 1, np.array([]), 0, 0)


def get_line(p1, p2, img, psd=70):
    """
    Search in image for neares most similar line of a given line.
    Search in H, S, and V for line in area arround given line. Select line
    with similiar rotation and many supporting edge pixel.

    Args:
        p1: Start point of user drawn line.
        p2: End point of user drawn line.
        img: Image in which neares line in searched.
        psd: Patch size divisor: Area arroung line in which similar line is 
             searched. Higher value = Smaller area

    Returns:
        [p1,p2]: List of two points of computed most similar line
    """

    # Get horizontal patch arround line and mask adapting to slope of line
    patch, mask, offset = get_patch(img, p1, p2, psd)

    # Offset points
    p1_o = p1 - offset
    p2_o = p2 - offset

    patch = cv2.cvtColor(np.uint8(patch), cv2.COLOR_BGR2HSV)
    patch = cv2.GaussianBlur(patch, (5,5), 0)

    best_lines = []
    for i in range(0,3):
        lines = line_detect(patch[:,:,i], mask)
        if lines is not None:
            best_lines.append(weight_lines(patch[:,:,i], lines, p1_o, p2_o))
    
    if len(best_lines) != 0:
        k = max(enumerate(best_lines), key=lambda x: x[1][2])[0]
        p1 = best_lines[k][0] + offset
        p2 = best_lines[k][1] + offset
    
    return [p1[0], p1[1], p2[0], p2[1]]

center_of_line = lambda p1, p2: np.array((np.float_((p1[0] + p2[0]))/2, np.float_((p1[1] + p2[1]))/2))
def get_transformed_patch(patch, p11, p12, p21, p22):
    delta = center_of_line(p11, p12) - center_of_line(p21, p22)

    #draw_line(patch, p21, p22)
    t = get_theta(p21, p22) * 180 / np.pi - 90 
    size = np.int_(np.sqrt(patch.shape[0]**2 + patch.shape[1]**2))
    patch_ = np.zeros((size,size,patch.shape[2]), dtype=np.float32)
    patch_[(size-patch.shape[0])/2: (size-patch.shape[0])/2 + patch.shape[0],
           (size-patch.shape[1])/2: (size-patch.shape[1])/2 + patch.shape[1],
           :] = patch

    rm = cv2.getRotationMatrix2D((size/2, size/2), t, 1)
    tm = np.array([[1,0,delta[0]],[0,1,delta[1]]], dtype=np.float32)
    patch = cv2.warpAffine(patch_, tm, (size, size))
    patch = cv2.warpAffine(patch, rm, (size, size))

    return patch


def get_corresponding_line(img1, img2, line1, psd=20, max_lines_to_check = 30):
    
    # dst = imge in which line is already found
    # src = image in which coresponding line is searched

    p11 = (line1[0], line1[1])
    p12 = (line1[2], line1[3])
    patch, mask, offset = get_patch(img1, p11, p12, psd)
    patch2, mask2, offset2 = get_patch(img2, p11, p12, psd)
    patch = cv2.cvtColor(patch[:,:,0:3], cv2.COLOR_BGR2HSV)
    patch2 = cv2.cvtColor(patch2[:,:,0:3], cv2.COLOR_BGR2HSV)
    p11 = p11 - offset
    p12 = p12 - offset

    # Make dst patch horizontal
    #draw_line(patch, p11, p12)
    t = get_theta(p11, p12) * 180 / np.pi - 90 
    size = np.int_(np.sqrt(patch.shape[0]**2 + patch.shape[1]**2))
    patch_ = np.zeros((size,size,patch.shape[2]), dtype=np.float32)
    patch_[(size-patch.shape[0])/2 : (size-patch.shape[0])/2 + patch.shape[0],
           (size-patch.shape[1])/2 : (size-patch.shape[1])/2 + patch.shape[1],
           :] = patch
    rm = cv2.getRotationMatrix2D((size/2, size/2), t, 1)
    patch = cv2.warpAffine(patch_, rm, (size, size))
    
    # Detect lines in HSV src image
    patch2_t = cv2.GaussianBlur(np.uint8(patch2), (5,5), 0)
    best_lines = []
    weights = []
    for i in range(0,3):
        lines = line_detect(patch2_t[:,:,i])
        if lines is not None:
            lines_, weights_ = weight_lines(patch2_t[:,:,i], lines, p11, p12, 10, return_best_line=False)
            best_lines.append(lines_)
            weights.append(weights_)

    # Compare lines in src image witch line in dst image  
    if len(weights) > 0:
        weights = np.array(weights).flatten()
        best_lines = np.array(best_lines).flatten()[np.argsort(weights)[::-1]]

        weights_t = np.empty((max_lines_to_check), dtype=np.float_)

        #TODO np.flip comes with numpy version 1.13
        for i in range(0, min(max_lines_to_check, len(weights))):
            patch2_t = get_transformed_patch(np.copy(patch2), p11, p12, tuple(best_lines[i][0]), tuple(best_lines[i][1]))[:,:,2]
            patch_t = np.copy(patch[:,:,2])
            patch2_t[patch_t==0] = 0
            patch_t[patch2_t==0] = 0
            patch2_t = cv2.equalizeHist(np.uint8(patch2_t))
            patch_t = cv2.equalizeHist(np.uint8(patch_t))
            patch2_t = cv2.GaussianBlur(patch2_t, (11,11), 0)
            patch_t = cv2.GaussianBlur(patch_t, (11,11), 0)
            
            #_, patch2_s = cv2.threshold(patch2_t, patch2_t.mean(dtype=np.float64), 255, cv2.THRESH_BINARY)
            #_, patch_s = cv2.threshold(patch_t, patch_t.mean(dtype=np.float64), 255, cv2.THRESH_BINARY)
            overlap = np.count_nonzero(patch_t)
#            _, patch2_s = cv2.threshold(np.uint8(patch2_t), 112, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#            _, patch_s = cv2.threshold(np.uint8(patch_t), 112, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            patch2_s = cv2.adaptiveThreshold(np.uint8(patch2_t), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0)
            patch_s = cv2.adaptiveThreshold(np.uint8(patch_t), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0)
            


            weights_t[i] = np.sum(abs(patch2_s - patch_s), dtype=np.int64)/overlap
            print overlap


           # cv2.imshow('sadfaeqw', np.uint8(patch_s ))
           # cv2.imshow('sadfaeqw2', np.uint8(patch2_s ))
           # cv2.waitKey(0)
            
            """
            #  summed orthagonnal image patch for matching
            patch2_s = np.sum(patch2_t, axis=1)/(np.count_nonzero(patch2_t, axis=1))
            patch_s = np.sum(patch_t, axis=1)/(np.count_nonzero(patch_t, axis=1))
            weights_t[i] = la.norm((patch2_s - patch_s)**2)
            patch_s = patch_s.reshape(-1,1)
            patch_ss = np.broadcast_to(patch_s, (len(patch_s),40))
            patch2_s = patch2_s.reshape(-1,1)
            patch2_ss = np.broadcast_to(patch2_s, (len(patch2_s),40))
            cv2.imshow('sadfaeqw', np.uint8(patch_ss * 255))
            cv2.imshow('sadfaeqw2', np.uint8(patch2_ss * 255))
            cv2.waitKey(0)
            """
            #weights_t[i] = la.norm(cv2.resize(patch2_t, (0,0), fx=0.1, fy=0.1) - cv2.resize(patch, (0,0), fx=0.1, fy=0.1)**2)
            print weights[i]
            #dpimage = np.zeros(patch.shape, patch.dtype)
            #dpimage[:,:,0] = patch_s
            #dpimage[:,:,1] = patch2_s
            #dpimage[:,:,2] = patch2_s
            #cv2.imshow('sdsf', np.uint8(dpimage))
#            cv2.imshow('bla', cv2.addWeighted(np.uint8(patch2_t[:,:,2]), 0.5, np.uint8(patch[:,:,2]), 0.5, 0))
#            cv2.waitKey(0)

        min_i = np.argmin(weights_t)
        return line1
        return np.array(best_lines[i]).flatten() + np.tile(offset2, 2)
        
    else:
        return None






