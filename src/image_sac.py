import cv2
import numpy as np
from image_helpers import lce

# scale factor for subimage size on which template matching is done
SUBIMAGE_SIZE_SCALE = 0.159  
# scale factor for template size
TEMPLATE_SIZE_SCALE = 0.07
# minimum size of user drawn rectangle to start point search else return middle point
MIN_RECT_SIZE = 5


def get_point_from_rectangle(img1, point1, point2):
    """Computes point of interest in a subimage which is defined by to given points."""

    assert 0 <= point1[1] < img1.shape[0], "Point1 outside image"
    assert 0 <= point1[0] < img1.shape[1], "Point1 outside image"
    assert 0 <= point2[1] < img1.shape[0], "Point2 outside image"
    assert 0 <= point2[0] < img1.shape[1], "Point2 outside image"
    
    #TODO replace for build with:
    #assert point1[0] != point2[0], "X cordinates of rectangle corners are equal -> no rectangle"
    #assert point1[1] != point2[1], "Y cordinates of rectangle corners are equal -> no rectangle"
    # if rectangle is to small return middlepoint of the two given points, assuming user 
    # wanted to select a single point and not draw rectangle
    if abs(point1[0] - point2[0]) < MIN_RECT_SIZE or abs(point1[1] - point2[1]) < MIN_RECT_SIZE:
        return (int((point1[0]+point2[0])/2), int((point1[1]+point2[1])/2))

    subimage = np.copy(img1[min(point1[1],point2[1]):max(point1[1],point2[1]), 
                                          min(point1[0], point2[0]):max(point1[0],point2[0])])
    subimage_gray = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)
    subimage_f = np.float32(subimage_gray)
    subimage_f = cv2.normalize(subimage_f, subimage_f, 0, 1, cv2.NORM_MINMAX)
    subimage_f = cv2.GaussianBlur(subimage_f, (5,5), 0)   
    
    # Detector parameters
    block_size = 2
    aperture_size = 3
    k = 0.04
    # Detecting corners
    corners = cv2.cornerHarris( subimage_f, block_size, aperture_size, k, cv2.BORDER_DEFAULT )

    # Assume that user wants to mark point in middle of rectangle, hence weight cornes using gaussian
    rows, cols = corners.shape
    gaus_cols = cv2.getGaussianKernel(cols, -1)
    gaus_rows = cv2.getGaussianKernel(rows, -1)
    gaus_matrix = gaus_rows*gaus_cols.T
    gaus_matrix_normalized = gaus_matrix/gaus_matrix.max()
    corners = corners * gaus_matrix_normalized
    
    # get sharpest corners
    i, j = np.where(corners == corners.max());

    # get index of corner in middle of sharpest corner array, most often there is only one entry 
    index = int(i.shape[0]/2)

    #add the start position of rectangle as offset
    return_point = (j[index] + min(point1[0], point2[0]), i[index] + min(point1[1], point2[1]))

    return return_point

def get_coresponding_point(img1, img2, point):
    """Search for coresponding point on second image given a point in first image using template matching."""

    assert 0 <= point[1] < img1.shape[0], "Point outside image 1. Have both images the same size?"
    assert 0 <= point[0] < img1.shape[1], "Point outside image 1. Have both images the same size?"
    assert 0 <= point[1] < img2.shape[0], "Point outside image 2. Have both images the same size?"
    assert 0 <= point[0] < img2.shape[1], "Point outside image 2. Have both images the same size?"

    # diameter of the meaningfull keypoint neighborhood 
    template_size = int((img1.shape[0] + img1.shape[1]) * 0.5  * TEMPLATE_SIZE_SCALE)

    # size of local subimage in which dense sampling is done
    subimage_size = int((img2.shape[0] + img2.shape[1]) * 0.5 * SUBIMAGE_SIZE_SCALE)

    # get template from img1 in which user draw
    template_size_half = int(template_size/2)
    x1 = max(point[0] - template_size_half, 0)
    x2 = min(point[0] + template_size_half, img1.shape[1] - 1)
    y1 = max(point[1] - template_size_half, 0)
    y2 = min(point[1] + template_size_half, img1.shape[0] - 1)
    subimage1 = np.copy(img1[y1:y2, x1:x2])

    # create subimage from img2 in which template is searched
    subimage_size_half = int(subimage_size/2)
    x1 = max(point[0] - subimage_size_half, 0)
    x2 = min(point[0] + subimage_size_half, img2.shape[1] - 1)
    y1 = max(point[1] - subimage_size_half, 0)
    y2 = min(point[1] + subimage_size_half, img2.shape[0] - 1)
    subimage2 = np.copy(img2[y1:y2, x1:x2])

    # preprocess both subimages
    subimage1F = np.float32(subimage1)
    subimage1F = lce(subimage1F, 11, 5)
    subimage1F = cv2.cvtColor(subimage1F, cv2.COLOR_BGR2GRAY)
    subimage1F = cv2.normalize(subimage1F, subimage1F, 0, 1, cv2.NORM_MINMAX)
    subimage1F = cv2.GaussianBlur(subimage1F, (5,5), 0) 
    subimage1X = cv2.Scharr(subimage1F, ddepth = -1, dx = 1, dy = 0)
    subimage1Y = cv2.Scharr(subimage1F, ddepth = -1, dx = 0, dy = 1)
    subimage1F = subimage1X + subimage1Y
    subimage1F[subimage1F < 0.5] = 1 - subimage1F[subimage1F < 0.5]
    #subimage1F = cv2.dilate(subimage1F, None)
    #subimage1F = cv2.erode(subimage1F, None)
    subimage1F = cv2.normalize(subimage1F, subimage1F, 0, 1, cv2.NORM_MINMAX)

    cv2.namedWindow("template", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("template", np.uint8(cv2.normalize(subimage1F, subimage1F, 0, 255, cv2.NORM_MINMAX)))
    cv2.resizeWindow("template", 640, 480)

    subimage2F = np.float32(subimage2)
    subimage2F = lce(subimage2F, 11, 5)
    subimage2F = cv2.cvtColor(subimage2F, cv2.COLOR_BGR2GRAY)
    subimage2F = cv2.normalize(subimage2F, subimage2F, 0, 1, cv2.NORM_MINMAX)
    subimage2F = cv2.GaussianBlur(subimage2F, (5,5), 0) 
    subimage2X = cv2.Scharr(subimage2F, ddepth = -1, dx = 1, dy = 0)
    subimage2Y = cv2.Scharr(subimage2F, ddepth = -1, dx = 0, dy = 1)
    subimage2F = subimage2X + subimage2Y
    subimage2F[subimage2F < 0.5] = 1 - subimage2F[subimage2F < 0.5]
    #subimage2F = cv2.dilate(subimage2F, None)
    #subimage2F = cv2.erode(subimage2F, None)
    subimage2F = cv2.normalize(subimage2F, subimage2F, 0, 1, cv2.NORM_MINMAX)

    cv2.namedWindow("subimage", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("subimage", np.uint8(cv2.normalize(subimage2F, subimage2F, 0, 255, cv2.NORM_MINMAX)))
    cv2.resizeWindow("subimage", 640, 480)

    # template matching
    # norms are missing in cv2 python wrapper
    CV_TM_SQDIFF = 0
    CV_TM_SQDIFF_NORMED = 1
    CV_TM_CCORR = 2
    CV_TM_CCORR_NORMED = 3
    CV_TM_CCOEFF = 4
    CV_TM_CCOEFF_NORMED = 5
    template_result = cv2.matchTemplate(subimage2F, subimage1F, CV_TM_CCOEFF_NORMED)
    template_result1 = cv2.normalize(template_result, template_result, 0, 1, cv2.NORM_MINMAX)

    cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("result", np.uint8(cv2.normalize(template_result1, template_result1, 0, 255, cv2.NORM_MINMAX)))
    cv2.resizeWindow("result", 640, 480)

    while cv2.waitKey(0) != 99:
        pass

    cv2.destroyWindow("subimage")
    cv2.destroyWindow("template")
    cv2.destroyWindow("result")

    min_val, max_val, min_loc , max_loc = cv2.minMaxLoc(template_result1)

    point2 = (max_loc[0] + template_size_half, max_loc[1] + template_size_half)

    return_point = (x1 + int(point2[0]), y1 + int(point2[1]))
    return return_point

def get_p_from_rectangle_a_coresponding_p(img1, img2, point1, point2):
    """Wrapper for getPoint_from_rectangle and get_coresponding_point.
    imageSelect: True if user draw rect on image 1, False if user draw on image 2.
    """
    return_point1 = get_point_from_rectangle(img1, point1, point2)
    return_point2 = get_coresponding_point(img1, img2, return_point1)
    #returnPoint2 = getPointFromRectangle(img2, (returnPoint1[0]-20,returnPoint1[1]-20), (returnPoint1[0]+20, returnPoint1[1]+20))
    return return_point1, return_point2
