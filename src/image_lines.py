import numpy as np
import cv2
from image_helpers import statistic_canny
from image_helpers import adaptive_thresh
from image_helpers import line_intersect 
from image_helpers import lce
from image_helpers import draw_line
import find_obj as f_o
import fast_match.fast_match_wrapper as f_m
from numpy.linalg import inv

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


def weight_lines(patch_p, lines, p1_o, p2_o):
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
    nl = min(10, lines.shape[0])
    line_img = np.empty(patch_p.shape, dtype=np.uint8)
    weights = np.empty((nl), dtype=np.float32)
    line_segs = np.empty((nl), dtype=object)
    i, j = 0, 0
    theta_o = get_theta(p1_o, p2_o)    

    while  j < lines.shape[0] and i < 10 :
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

    if i > 0:
        k = weights[:i].argmax()
        p1 = line_segs[k][0] 
        p2 = line_segs[k][1] 
        w = weights[k]
    else:
        p1 = p1_o
        p2 = p2_o
        w = 0

    return p1, p2, w


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
    sigma = 0.33
    magic_n = 100
    for i in range(0,3):
        ch = cv2.equalizeHist(patch[:,:,i])

        # Edge dedection
        m = np.median(ch)
        lower_bound = int(max(0, (1.0 - sigma) * m)) + magic_n
        upper_bound = int(min(255, (1.0 + sigma) * m)) + magic_n
        ch = cv2.Canny(ch, lower_bound, upper_bound, L2gradient=True)

        # Filter patch_p to only contain the patch area not the containing rectangle
        ch = ch * np.uint8(mask)
        
        # Line detection
        lines = cv2.HoughLines(ch, 1, np.pi/180.0, 1, np.array([]), 0, 0)

        if lines is not None:
            best_lines.append(weight_lines(ch, lines, p1_o, p2_o))
    
    if len(best_lines) != 0:
        k = max(enumerate(best_lines), key=lambda x: x[1][2])[0]
        p1 = best_lines[k][0] + offset
        p2 = best_lines[k][1] + offset
    
    return [p1[0], p1[1], p2[0], p2[1]]


def get_corresponding_line(img1, img2, line1, eng):
    
    template, t_mask, t_offset = get_patch(img1[:,:,0:3], (line1[0], line1[1]), 
                                           (line1[2], line1[3]), psd=30)

    image, i_mask, i_offset = get_patch(img2[:,:,0:3], (line1[0], line1[1]),
                                        (line1[2], line1[3]), psd=10)

    s = 300
    sf = s / float(np.amax(image.shape))
    image = cv2.resize(image, (0,0), fx=sf, fy=sf) 
    template = cv2.resize(template, (0,0), fx=sf, fy=sf) 

    template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)[:,:,2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2]
    template = cv2.equalizeHist(np.uint8(template))
    image = cv2.equalizeHist(np.uint8(image))
    
    trans_mat = f_m.get_template_location(image, template, eng)

    print(trans_mat)
    line_t = np.float32(line1).reshape(1,-2,2)
    print(line1)
    line2 = cv2.perspectiveTransform(line_t, inv(trans_mat)).reshape(4).tolist()
    print(line2)

##    template = template * np.reshape(t_mask, (t_mask.shape[0], t_mask.shape[1], 1))
##    image = image * np.reshape(i_mask, (i_mask.shape[0], i_mask.shape[1], 1))
#
#    template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#    
#    
#    #result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#    #result = np.abs(result)**3                                           
#    #val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)      
#    #result8 = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U) 
#    #cv2.imshow("result", result8)   
#    #cv2.imshow("template", np.uint8(template))
#    #cv2.imshow("image", np.uint8(image))
#
#    feature_name = 'surf-flann'
#
#
#    detector, matcher = f_o.init_feature(feature_name)
#
#
#    if detector is None:
#        print('unknown feature:', feature_name)
#        sys.exit(1)
#
#    print('using', feature_name)
#
#
#    #image = cv2.GaussianBlur(cv2.equalizeHist(np.uint8(image[:,:,2])), (5,5),0)
#    #template = cv2.GaussianBlur(cv2.equalizeHist(np.uint8(template[:,:,2])), (5,5), 0)
#    #image = cv2.GaussianBlur((np.uint8(image[:,:,2])), (5,5),0)
#    #template = cv2.GaussianBlur((np.uint8(template[:,:,2])), (5,5), 0)
#    #image = np.uint8(image[:,:,1])
#    #image = cv2.equalizeHist(image)
#    #template = np.uint8(template[:,:,1])
#    #cv2.imshow('asdr', template)
#    image = cv2.normalize(image[:,:,2], None,0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#    template = cv2.normalize(template[:,:,2], None,0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#
#
#    s = 200
#    sf = s / float(np.amax(image.shape))
#    image = cv2.resize(image, (0,0), fx=sf, fy=sf) 
#    template = cv2.resize(template, (0,0), fx=sf, fy=sf) 
#    
#    cv2.imwrite(str(line1[0])+'t.ppm', template)
#    cv2.imwrite(str(line1[0])+'i.jpg', image)
#
#    kp1, desc1 = detector.detectAndCompute(template, None)
#    kp2, desc2 = detector.detectAndCompute(image, None)
#    
#    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
#
#    def match_and_draw(win):
#        print('matching...')
#        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
#        p1, p2, kp_pairs = f_o.filter_matches(kp1, kp2, raw_matches, ratio = 1.4)
#        if len(p1) >= 4:
#            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
#            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
#        else:
#            H, status = None, None
#            print('%d matches found, not enough for homography estimation' % len(p1))
#
#        vis = f_o.explore_match(win, template, image, kp_pairs, status, H)
#
#    match_and_draw('find_obj')
#    cv2.waitKey()

    return line2
