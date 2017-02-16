from __future__ import print_function
import numpy as np
import cv2
#from image_pst import pst
from ler.image_ler import max_size


pint = lambda p: (int(p[0]), int(p[1]))

vprint = lambda *a, **k: None
def set_verbose(verbose):
    """
    Sets the global verbosity function.

    Args:
        verbose: If true verbose output, false no verbose output.
    """
    if verbose:
        def _vprint(*args, **kwargs):
            print(*args , **kwargs)
    else:   
        _vprint = lambda *a, **k: None     
    global vprint
    vprint = _vprint


def draw_frame(img, x_min, x_max, y_min, y_max):
    """
    Draws a frame on a given image.
    Used to display cropping lines

    Args:
        img: Image on which frame is drawn
        x_min: X coordinate of smaller point of rectangle
        y_min: Y coordinate of smaller point of rectangle
        x_max: X coordinate of bigger point of rectangle
        y_max: Y coordinate of bigger point of rectangle
    """
    thickness = int((img.shape[0] + img.shape[1]) / 900  ) + 1
    lineType = 8
    color = (255,255,255)
    cv2.line(img, (x_min, y_min), (x_min, y_max), color, thickness, lineType )
    cv2.line(img, (x_min, y_max), (x_max, y_max), color, thickness, lineType )
    cv2.line(img, (x_max, y_max), (x_max, y_min), color, thickness, lineType )
    cv2.line(img, (x_max, y_min), (x_min, y_min), color, thickness, lineType )


def draw_line(img, start, end, color=(255,255,255), l_number=-1):
    """
    Draws line and line number on given image.
    
    Args:
        img: Image to be drawn on.
        start: Startpoint of line.
        end: Endpoint of line.
        color: Color of the line. If no color given line is white.
        l_number: Linenumber. If no number given only line is drawn.
    """
    thickness = int((img.shape[0] + img.shape[1]) / 900  ) + 1
    lineType = 8
    cv2.line(img, pint(start), pint(end), color, thickness, lineType )
    if l_number > 0:
        text_size = float(thickness)/2 
        cv2.putText(img, str(l_number), pint(end), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, text_size, (0,255,0), thickness)


def draw_rectangle(img, start, end, color=(255,255,255)):
    """
    Draws rectangle on given image.
    
    Args:
        img: Image to be drawn on.
        start: Startpoint of rectangle.
        end: Endpoint of rectangle.
        color: Color of the line. If no color given line is white.
    """
    thickness = int((img.shape[0] + img.shape[1]) / 900  ) + 1
    lineType = 8
    cv2.rectangle(img, start, end, color, thickness, lineType )


def draw_circle(img, center, color=(255,255,255)):
    """
    Draws circle on given image.
    
    Args:
        img: Image to be drawn on.
        center: Center of circle
        color: Color of the line. If no color given line is white.
    """
    radius = int((img.shape[0] + img.shape[1]) / 400 ) + 1
    linetype = -1
    cv2.circle(img, (int(center[0]), int(center[1])), radius, color, linetype)


def weighted_average_point(point1, point2, alpha):
    """
    Return the average point between two points weighted by alpha.
    
    Args:
        point1: First point multiplied by 1- alpha.
        point2: Second point multiplied by alpha.

    Returns:
        point: Weighted point
    """
    x = int((1 - alpha) * point1[0] + alpha * point2[0])
    y = int((1 - alpha) * point1[1] + alpha * point2[1])
    return (x,y)


def lce(img, kernel = 11 , amount = 0.5):
    """
    Local Contrast Enhancement by unsharp mask.
    From the value channel of the image in hsv color space a gaussian blured
    version is subtracted.

    Args:
        img: BGR-Image which is enhanced
        kernel: Size of the gaussian kernel.
        amount: Strength of the contrast enhancment.

    Returns:
        img_bgr: Contrast enhanced np.float32 BGR-Image, values between 0, 255.
    """
    
    assert kernel % 2 == 1, "kernel size has to be odd."

    img = np.float32(img)
    img = cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_v = img_hsv[:,:,2] 
    img_v_blurred = cv2.GaussianBlur(img_v, (kernel, kernel), 0, 0)
    img_t = img_v - img_v_blurred
    img_v = img_v + img_t * amount
    
    img_v[img_v > 1] = 1
    img_v[img_v < 0] = 0

    img_hsv[:,:,2] = img_v

    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img_bgr*255


def unsharp_mask(img, sigma=1, amount=0.8):
    """
    Sharpends given image via unsharp mask.

    Args:
        img: Image to be sharpened.
        sigma: Sigma of Gaussian kernel for bluring.
        amount: Amount of sharpening.

    Returns:
        img: Sharpened Image.
    """

    img_blured = cv2.GaussianBlur(img, (0,0), sigma)
    img = img + (img - img_blured) * amount
    img[img > 255] = 255
    img[img < 0] = 0

    return img


def get_crop_idx(crop_img, scale = 400):
    """
    Computes crop indices based on alpha channel.
    Searches biggest white rectangle in alpha channel.

    Args:
        crop_img: to be cropped image with alpha channel
        scale: Downsample image by img size / scale to speed up

    Returns:
        idx: Cropindices [x_min, y_min, x_max, y_max]
    """

    # Speed up by downsmpling the crop image costs accuracy of crop indices
    ac = int(np.sum(crop_img.shape)/scale)
    return  max_size(crop_img[::ac,::ac], 2) * ac + [ac, ac, -ac, -ac]


def scale_image_lines_points(img, lines, points, scale_factor):
    """
    Scales an image, points and lines in this image by a given scale factor.

    Args:
        img: To be scaled image.
        lines: To be scaled lines.
        points: To be scaled points.
        scale_factor: Factor by which image, lines and points are scaled.

    Returns:
        img: Scaled image.
        lines: Scaled lines.
        points: Scaled points.
    """
    scale_lines = lambda ls, f: [[v * f for v in l] for l in ls]   
    scale_points = lambda ps, f: [tuple([pc * f for pc in p]) for p in ps]

    if scale_factor > 1:
        # INTER_CUBIC is bugy with bw images
        #method = cv2.INTER_CUBIC
        method = cv2.INTER_LINEAR
    else:
        method = cv2.INTER_AREA

    img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor,
          interpolation=method)

    lines = scale_lines(lines, scale_factor)
    points = scale_points(points, scale_factor)

    return img, lines, points
    

def do_scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
    scale_img1, scale_img2, scale_factor):
    """
    Scales an image, line and point pair.
    Uses a scale factor per image and one global factor.

    Args:
        img1: First image to be scaled.
        img2: Second image to be scaled.
        lines_img1: First list of lines to be scaled.
        lines_img2: Second list of lines to be scaled.
        points_img1: First list of points to be scaled.
        points_img2: Second list of points to be scaled.
        scale_img1: Scale factor for first image/lines/points.
        scale_img2: Scale factor for second image/lines/points.
        scale_factor: global scale factor.

    Returns:
        img1: Scaled first image.
        img2: Scaled second image.
        lines_img1: Scaled first list of lines.
        lines_img2: Scaled second list of lines.
        points_img1: Scaled first list of points.
        points_img2: Scaled second list of points.
        scale_img1: Scale factor of first image times global scale factor.
        scale_img2: Scale factor of second image times global scale factor.
    """
    scale_img1 *= scale_factor
    scale_img2 *= scale_factor
    
    if scale_img1 != 1:
        img1, lines_img1, points_img1 = scale_image_lines_points(img1, lines_img1, points_img1, scale_img1)

    if scale_img2 != 1:
        img2, lines_img2, points_img2  = scale_image_lines_points(img2, lines_img2, points_img2, scale_img2)

    return img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
           scale_img1, scale_img2
        

def scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
    scale_factor=1):
    """
    Upscales the smaller image and coresponding lines/points of two given images.
    If scale factor is given all entities are scaled by this factor.
    Aspect ratio is preserved, blank space is filled with zeros.
    Args:
        img1: Image 1.
        img2: Image 2.
        lines_img_1: Lines in image 1.
        lines_img_2: Lines in image 2.
        points_img_1: Points in image 1.
        points_img_2: Points in image 2.
        scale_factor: Scaling factor for image/line/point pair. 

    Returns:
        img1: If img1 is bigger returns img1 else scaled img1.
        img2: If img2 is bigger returns img2 else scaled img2.
        lines_img_1: Lines in image 1, scaled if img1 is scaled.
        lines_img_2: Lines in image 2, scaled if img2 is scaled.
        points_img_1: Points in image 1, scaled if img1 is scaled.
        points_img_2: Points in image 2, scaled if img2 is scaled.
        scale_factor_img1: Scale factor by which first image/lines/points were
            scaled.
        scale_factor_img1: Scale factor by which second image/lines/points were
            scaled.
        x_max: After x_max one image is padded with zeros in x direction.
        y_max: After y_max one image is padded with zeros in y direction.

    """
    y_size_img1, x_size_img1, z_size_img1 = img1.shape
    y_size_img2, x_size_img2, z_size_img2 = img2.shape
    x_scale_factor = float(x_size_img1)/ float(x_size_img2)
    y_scale_factor = float(y_size_img1)/ float(y_size_img2)

   

    # Images are of same size
    if x_size_img1 == x_size_img2 and y_size_img1 == y_size_img2:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
            scale_factor_img1, scale_factor_img2 = do_scale(img1, img2,
            lines_img1, lines_img2, points_img1, points_img2, 1, 1, scale_factor)
        x_max = img1.shape[1]
        y_max = img1.shape[0]

    # Image 1 is bigger
    elif x_size_img1 >= x_size_img2 and y_size_img1 >= y_size_img2:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
            scale_factor_img1, scale_factor_img2 = do_scale(img1, img2, 
            lines_img1, lines_img2, points_img1, points_img2, 1, 
            min(x_scale_factor, y_scale_factor), scale_factor)
        temp_img = np.zeros_like(img1)
        temp_img[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]] = img2
        x_max = img2.shape[1]
        y_max = img2.shape[0]
        img2 = temp_img

    # Image 1 is smaller
    elif x_size_img1 <= x_size_img2 and y_size_img1 <= y_size_img2:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
            scale_factor_img1, scale_factor_img2 = do_scale(img1, img2, 
            lines_img1, lines_img2, points_img1, points_img2,  
            1/max(x_scale_factor, y_scale_factor), 1, scale_factor)
        temp_img = np.zeros_like(img2)
        temp_img[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]] = img1
        x_max = img1.shape[1]
        y_max = img1.shape[0]
        img1 = temp_img

    # Images size relations are not the same i.e. x_scale < 1 and y_scale > 1 
    # or vice versa
    else:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
            scale_factor_img1, scale_factor_img2 = do_scale(img1, img2, 
            lines_img1, lines_img2, points_img1, points_img2, 1, 1, scale_factor)
        temp_img = np.zeros((max(img1.shape[0], img2.shape[0]), 
            max(img1.shape[1], img2.shape[1]), max(img1.shape[2], img2.shape[2])), 
            dtype=img1.dtype)
        temp_img2 = np.copy(temp_img)
        x_max = min(img1.shape[1], img2.shape[1])
        y_max = min(img1.shape[0], img2.shape[0])
        temp_img[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]] = img1
        img1 = temp_img
        temp_img2[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]] = img2
        img2 = temp_img2

    return img1, img2, lines_img1, lines_img2, points_img1, points_img2,\
        scale_factor_img1, scale_factor_img2, x_max, y_max


def adaptive_thresh(img):
    """
    Thresholds given image adaptive.

    Args:
        img: Image to be thresholded.

    Returns:
        img: Thresholded grey image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 10)


'''
def pst_wrapper(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.GaussianBlur(img, (5,5), 0)
    #pst_img = abs(pst(img, morph_flag=True))
    #pst_img =  np.uint8((pst_img/pst_img.max())*255)
    img = cv2.resize(img, (0,0), fx=2, fy=2)
    pst_img = pst(img, morph_flag=True) 

    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    h_m = cv2.morphologyEx(pst_img, cv2.MORPH_OPEN, struct_elem, iterations=1)
    h_m = cv2.erode(pst_img, struct_elem, iterations=1)
    #print h_m.max()
    cv2.imshow('h_m', h_m * 255)
    pst_cleaned = pst_img * (1-h_m)
    #print pst_img.max()
    #pst_img = cv2.morphologyEx(pst_img, cv2.MORPH_GRADIENT, (3,3))
    cv2.imshow('cleaned', pst_cleaned)
    pst_img = cv2.resize(pst_img, (0,0), fx=0.5, fy=0.5)
    return np.uint8(pst_img* 255)
'''

def statistic_canny(img, sigma=0.33):
    """
    Edge detection on color image depending on image properties.
    
    Args:
        img: Image on which to detect edges.
        sigma: Standard deviation.

    Returns:
        img: Edge image.
    """

    img = cv2.GaussianBlur(img, (5,5), 0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    m = np.median(img)

    lower_bound = int(max(0, (1.0 - sigma) * m))
    upper_bound = int(min(255, (1.0 + sigma) * m))

    h = cv2.Canny(img[:,:,0], lower_bound, upper_bound, L2gradient=True)
    s = cv2.Canny(img[:,:,1], lower_bound, upper_bound, L2gradient=True)
    v = cv2.Canny(img[:,:,2], lower_bound, upper_bound, L2gradient=True)
    
    return(np.maximum(h, np.maximum(s, v)))  


def line_intersect(a1, a2, b1, b2) :
    """
    Compute intersection of two lines.
    All input points have to be float.
    Returns startpoint of second line if lines are parallel.
    Args:
        a1: Startpoint first line.
        a2: Endpoint first line.
        b1: Startpoint second line.
        b2: Endpoint second line.

    Returns:
        intersection point.
    """
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = np.empty_like(da)
    dap[0] = -da[1]
    dap[1] = da[0]
    denom = np.dot(dap, db)
    if denom == 0:
        return b1
    num = np.dot(dap, dp)
    return (num / denom) * db + b1


def show_image(img, name='img', x=1000, y=1000):
    """
    Resizes image and displays it in window with given name.
    Constraints both sides of image by given length constraints.

    Args:
        img: Image to be displayed.
        name: Name of openCV window in which image is displayed.
        x: Max size of image in x direction.
        y: Max size of image in y direction.
    """
    iy, ix, _ = img.shape
    
    x_scale = x/float(ix)
    y_scale = y/float(iy)

    if x_scale > y_scale:
        img_d = cv2.resize(np.uint8(img[:,:,0:3]), (0,0), fx=y_scale, fy=y_scale)
        scale = y_scale
    else:
        img_d = cv2.resize(np.uint8(img[:,:,0:3]), (0,0), fx=x_scale, fy=x_scale)
        scale = x_scale
    
    cv2.namedWindow(name ,cv2.WINDOW_NORMAL)
    cv2.imshow(name, img_d)
    cv2.resizeWindow(name, x,y)
    return img_d, scale
