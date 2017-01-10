import numpy as np
import cv2
from image_pst import pst
from ler.image_ler import max_size

def draw_line(img, start, end, color, l_number=-1):
    thickness = int((img.shape[0] + img.shape[1]) / 900  ) + 1
    lineType = 8
    cv2.line(img, start, end, color, thickness, lineType )
    if l_number > 0:
        text_size = float(thickness)/2 
        cv2.putText(img, str(l_number), end, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, text_size, (0,255,0), thickness)


def draw_rectangle(img, start, end, color):
    thickness = int((img.shape[0] + img.shape[1]) / 900  ) + 1
    lineType = 8
    cv2.rectangle(img, start, end, color, thickness, lineType )


def draw_circle(img, center, color):
    radius = int((img.shape[0] + img.shape[1]) / 400 ) + 1
    linetype = -1
    cv2.circle(img, center, radius, color, linetype)


def lce(img, kernel = 11 , amount = 0.5):
    """
    Local Contrast Enhancement by unsharp mask.
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

#def get_crop_idx(y_p, grid_shape, img_shape, x_max, y_max, scale = 400):
def get_crop_idx(crop_img, scale = 400):
    """
    Computes crop indices based on deformed mesh 

    Args:
        y_p: Point coordinates of mesh.
        grid_shape: Number of rows and columns in grid
        img_shape: Size of image to be cropped
        x_max: Maximum x value for right crop index
        y_max: Maximum y value for left crop index
        scale: Downsample image by img size / scale to speed up

    Returns:
        idx: Cropindices [x_min, y_min, x_max, y_max]
    """

    # Get outline points from grid
    #left = y_p[0:grid_shape[0]]
    #right = y_p[-grid_shape[0]:]
    #bottom = y_p[grid_shape[0]-1::grid_shape[0]]
    #top = y_p[::grid_shape[0]]

    # Create crop image
    #pp = np.int32(np.vstack([left, bottom, right[::-1], top[::-1]]))#.clip(min=0)
    #pp[:,0] = pp[:,0].clip(max=x_max)
    #pp[:,1] = pp[:,1].clip(max=y_max)
    #crop_image = np.zeros((img_shape[0], img_shape[1]), np.uint8)
    #cv2.fillPoly(crop_image, [pp], (1))
    #cv2.imshow('afg', crop_image * 255)

    # Speed up by downsmpling the crop image costs accuracy of crop indices
#    cv2.imshow('adf', crop_img)
    #print crop_img
    ac = int(np.sum(crop_img.shape)/scale)
    return  max_size(crop_img[::ac,::ac], 2) * ac + [ac, ac, -ac, -ac]
    #return max_size(crop_image, 1)


#def get_crop_indices(img):
#    """
#    Get crop indices to crop black border from image.
#    Crops non linear borders.
#    Starts with small rectangle in middle, grows till black pixels are reached
#    at each site. This inefficient method has to be used since morphing
#    does not necessarily produce straight edges.
#
#    Args: 
#        img1: BGRA image. Alpha channel is used to determine crop indices.
#
#    Returns:
#        x_min:
#        x_max:
#        y_min:
#        y_max:
#
#    """        
#    
#    x_min= int(img.shape[1] / 2) - 1
#    y_min= int(img.shape[0] / 2) - 1
#    x_max= int(img.shape[1] / 2) + 1
#    y_max= int(img.shape[0] / 2) + 1
#    
#    x_step_size = int((img.shape[1]/300) * 2)
#    y_step_size = int((img.shape[0]/300) * 2) 
#
#    top_done = False
#    left_done = False
#    bottom_done = False
#    right_done = False
#
#    while not top_done or not right_done or not bottom_done or not left_done:
#        # check left  
#        if not left_done and x_min - x_step_size > -1: 
#            for y in range(y_min, y_max):
#                if  img[y, x_min - x_step_size, 3] == 0:
#                    left_done = True   
#                    break
#            if y == y_max -1 :
#                left_done = False
#                x_min -= x_step_size
#        else:
#            left_done = True
#
#        # check bottom
#        if not bottom_done and y_max + y_step_size < img.shape[0] :
#            for x in range(x_min, x_max):
#                if img[y_max + y_step_size, x, 3] == 0:
#                    bottom_done = True 
#                    break
#            if x == x_max - 1:
#                bottom_done = False 
#                y_max += y_step_size
#        else:
#            bottom_done = True
#
#        # check right
#        if not right_done and x_max + x_step_size < img.shape[1] :
#            for y in range(y_min, y_max):
#                if img[y, x_max + x_step_size, 3] == 0:
#                    right_done = True
#                    break
#            if y == y_max - 1:
#                right_done = False
#                x_max += x_step_size
#        else:
#            right_done = True
#
#        # check top  
#        if not top_done and y_min - y_step_size > -1:
#            for x in range(x_min, x_max):
#                if img[y_min - y_step_size, x, 3] == 0:
#                    top_done = True
#                    break
#            if x == x_max - 1:
#                top_done = False
#                y_min -= y_step_size
#        else:
#            top_done = True
#
#    return x_min, x_max, y_min, y_max


def scale_image_lines_points(img, lines, points, scale_factor):

    scale_lines = lambda ls, f: [[v * f for v in l] for l in ls]   
    scale_points = lambda ps, f: [[pc * f for pc in p] for p in ps]

    if scale_factor > 1:
        method = cv2.INTER_CUBIC
    else:
        method = cv2.INTER_AREA

    img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor,
          interpolation=method)

    lines = scale_lines(lines, scale_factor)
    points = scale_points(points, scale_factor)

    return img, lines, points
    

def do_scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2, scale_img1, scale_img2, 
    scale_factor):
    scale_img1 *= scale_factor
    scale_img2 *= scale_factor

    print(scale_img1, scale_img2)
    if scale_img1 != 1:
        img1, lines_img1, points_img1 = scale_image_lines_points(img1, lines_img1, points_img1, scale_img1)

    if scale_img2 != 1:
        img2, lines_img2, points_img2  = scale_image_lines_points(img2, lines_img2, points_img2, scale_img2)

    return img1, img2, lines_img1, lines_img2, points_img1, points_img2
        

def scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2, scale_factor=1):
    """
    Upscales the smaller image and coresponding lines of two given images.
    Aspect ratio is preserved, blank space is filled with zeros.
    Args:
        img1: Image 1.
        img2: Image 2.
        lines_img_1: Lines in image 1.
        lines_img_2: Lines in image 2.
        scale_factor: Scaling factor for first image and both line lists. 

    Returns:
        img1: If img1 is bigger returns img1 else scaled img1.
        img2: If img2 is bigger returns img2 else scaled img2.
        lines_img_1: Lines in image 1, scaled if img1 is scaled.
        lines_img_2: Lines in image 2, scaled if img2 is scaled.
        x_max: After x_max one image is padded with zeros in x direction.
        y_max: After y_max one image is padded with zeros in y direction.

    """
    y_size_img1, x_size_img1, z_size_img1 = img1.shape
    y_size_img2, x_size_img2, z_size_img2 = img2.shape
    x_scale_factor = float(x_size_img1)/ float(x_size_img2)
    y_scale_factor = float(y_size_img1)/ float(y_size_img2)

   

    # Images are of same size
    if x_size_img1 == x_size_img2 and y_size_img1 == y_size_img2:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2 = do_scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2, 1, 1, scale_factor)
        x_max = img1.shape[1]
        y_max = img1.shape[0]

    # Image 1 is bigger
    elif x_size_img1 >= x_size_img2 and y_size_img1 >= y_size_img2:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2 = do_scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2, 1, 
            min(x_scale_factor, y_scale_factor), scale_factor)
        temp_img = np.zeros_like(img1)
        temp_img[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]] = img2
        x_max = img2.shape[1]
        y_max = img2.shape[0]
        img2 = temp_img

    # Image 1 is smaller
    elif x_size_img1 <= x_size_img2 and y_size_img1 <= y_size_img2:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2 = do_scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2,  
            1/max(x_scale_factor, y_scale_factor), 1, scale_factor)
        temp_img = np.zeros_like(img2)
        temp_img[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]] = img1
        x_max = img1.shape[1]
        y_max = img1.shape[0]
        img1 = temp_img

    # Images size relations are not the same i.e. x_scale < 1 and y_scale > 1 
    # or vice versa
    else:
        img1, img2, lines_img1, lines_img2, points_img1, points_img2 = do_scale(img1, img2, lines_img1, lines_img2, points_img1, points_img2, 1, 1, scale_factor)
        temp_img = np.zeros((max(y_size_img1, y_size_img2), 
            max(x_size_img1, x_size_img2), max(z_size_img1, z_size_img2)), 
            dtype=img1.dtype)
        temp_img2 = np.copy(temp_img)
        x_max = min(img1.shape[1], img2.shape[1])
        y_max = min(img1.shape[0], img2.shape[0])
        temp_img[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]] = img1
        img1 = temp_img
        temp_img2[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]] = img2
        img2 = temp_img2

    return img1, img2, lines_img1, lines_img2, points_img1, points_img2, x_max, y_max


def adaptive_thresh(img):
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
    Edge detection depending on image properties.
    
    Args:
        img: Image on which to detect edges.
        sigma: Standard deviation.

    Returns:
        img: Edged image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    m = np.median(img)

    lower_bound = int(max(0, (1.0 - sigma) * m))
    upper_bound = int(min(255, (1.0 + sigma) * m))

    return cv2.Canny(img, lower_bound, upper_bound, L2gradient=True)


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
