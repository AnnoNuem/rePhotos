#!/usr/bin/env python
"""Functions to find interesting points near user drawn points as well as
corresponding points.
"""
import cv2
import numpy as np
from image_gabor import get_gabor

# scale factor for subimage size on which template matching is done
SUBIMAGE_SIZE_SCALE = 0.3 #0.159  
# scale factor for template size
TEMPLATE_SIZE_SCALE = 0.07
# minimum size of user drawn rectangle to start point search else return middle
# point
MIN_RECT_SIZE = 5


def getPointFromRectangle(img1, point1, point2):
    """ Computes point of interest in a rectangle defined by two given points.
    
    The best corner is returned weighted by the distance to the center of the rectangle.
    
    Parameters
    ----------
    img1 : ndarray
        Image in which corner  point of interest is searched.
    point1 : tuple
        First corner of rectangle.
    point2 : tuple
        Opposite corner of first corner of rectangle.
    
    Returns
    -------
    returnPoint : tuple
        The best point of interest.

    """

    assert 0 <= point1[1] < img1.shape[0], "Point1 outside image"
    assert 0 <= point1[0] < img1.shape[1], "Point1 outside image"
    assert 0 <= point2[1] < img1.shape[0], "Point2 outside image"
    assert 0 <= point2[0] < img1.shape[1], "Point2 outside image"
    
    # if rectangle is to small return middlepoint of the two given points, 
    # assuming user wanted to select a single point and not draw rectangle
    if abs(point1[0] - point2[0]) < MIN_RECT_SIZE or abs(point1[1] - point2[1])\
        < MIN_RECT_SIZE:
        return (int((point1[0]+point2[0])/2), int((point1[1]+point2[1])/2))

    subimage = np.copy(img1[int(min(point1[1],point2[1])):\
                            int(max(point1[1],point2[1])), 
                            int(min(point1[0], point2[0])):\
                            int(max(point1[0],point2[0]))])
    subimageGray = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)
    subimageF = np.float32(subimageGray)
    subimageF = cv2.normalize(subimageF, subimageF, 0, 1, cv2.NORM_MINMAX)
    subimageF = cv2.GaussianBlur(subimageF, (5,5), 0)   
    
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    corners = cv2.cornerHarris( subimageF, blockSize, apertureSize, k,\
        borderType=cv2.BORDER_DEFAULT )

    # Assume that user wants to mark point in middle of rectangle, 
    # hence weight cornes using gaussian
    rows, cols = corners.shape
    gausCols = cv2.getGaussianKernel(cols, -1)
    gausRows = cv2.getGaussianKernel(rows, -1)
    gausMatrix = gausRows*gausCols.T
    gausMatrixNormalized = gausMatrix/gausMatrix.max()
    corners = corners * gausMatrixNormalized
    
    # get sharpest corners
    i, j = np.where(corners == corners.max());

    # get index of corner in middle of sharpest corner array, most often there 
    # is only one entry 
    index = int(i.shape[0]/2)

    #add the start position of rectangle as offset
    returnPoint = (j[index] + min(point1[0], point2[0]), i[index] +\
        min(point1[1], point2[1]))

    return returnPoint


def get_and_pre_patch(img, point, size_half):
    """Get a patch from an image and preprocess it.
    
    Algorithm returns biggest possible template limited by the image size. 

    Parameters
    ----------
    img : ndarray
        Image from which to get the patch.
    point: ndarray
        Centerpoint of the patch.
    size_half : int
        Half of the patchsize.

    Returns
    -------
    subimage : ndarray
        Template as copy.
    ndarray
        Offset of lowest corner of patch with respect to image origin.
    ndarray
        Differences between theorethical template size and practical
        template size limited by size of the image.

    """
    
    x1 = max(point[0] - size_half, 0)
    x2 = min(point[0] + size_half, img.shape[1])
    y1 = max(point[1] - size_half, 0)
    y2 = min(point[1] + size_half, img.shape[0])
    delta_x_min = point[0] - x1
    delta_x_max = x2 - point[0]
    delta_y_min = point[1] - y1
    delta_y_max = y2 - point[1]
    subimage = cv2.cvtColor(np.float32(img[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV).copy()
    subimage = cv2.normalize(subimage, subimage, 0.0, 1.0 , cv2.NORM_MINMAX)
    return subimage, np.array((x1, y1)), np.array([delta_x_min, delta_x_max, delta_y_min, delta_y_max], dtype=np.float32)   


def getCorespondingPoint(img1, img2, point, template_size_s=101):
    """Search for coresponding point on second image given a point in first image.

    First possible matching corners are searched, then template matching at the
    possible corner locations is done.

    Parameters
    ----------
    img1 : ndarray
        Image in which point was found.
    img2 : ndarray
        Image in which corresponding point is searched.
    point : ndarray
        Location of found point in img1.
    template_size_1 : int 
        Size of scaled template for template matching.

    Returns
    -------
    point: ndarray
        The corresponding point.

    """

    if not (0 <= point[1] < img1.shape[0]) or\
       not (0 <= point[0] < img1.shape[1]) or\
       not (0 <= point[1] < img2.shape[0]) or\
       not (0 <= point[0] < img2.shape[1]):
        return None

    point = np.array(point, dtype=np.int_)
    
    # diameter of the meaningfull keypoint neighborhood 
    templateSizeHalf = int(((img1.shape[0] + img1.shape[1]) * 0.5  * \
                              TEMPLATE_SIZE_SCALE)/2)

    # size of local subimage in which dense sampling is done
    subimageSizeHalf = int(((img2.shape[0] + img2.shape[1]) * 0.5 * \
                             SUBIMAGE_SIZE_SCALE)/2)

    # get template from img1 in which user draw
    subimageF1, _, deltas = get_and_pre_patch(img1, point, templateSizeHalf)

    # create subimage from img2 in which template is searched
    subimageF2, offset, _ = get_and_pre_patch(img2, point, subimageSizeHalf)

    # get possible corners
    corners = cv2.goodFeaturesToTrack(np.uint8(subimageF2[...,2]*255), 2000,\
        qualityLevel=0.01, minDistance=10)

    # Scale patches, lines and points
    sf = template_size_s/np.float32(np.max(subimageF1.shape[0:2]))
    subimageF2 = cv2.resize(subimageF2, (0,0), fx=sf, fy=sf)
    subimageF1 = cv2.resize(subimageF1, (0,0), fx=sf, fy=sf)
    corners_s = [corner[0] * sf for corner in corners]
    deltas = np.int_(deltas * sf)
    weights = np.zeros((corners.shape[0]), dtype=np.float32)
    i = 0

    subimageF2 = cv2.normalize(get_gabor(subimageF2)[...,2], subimageF2, 0, 1,\
        cv2.NORM_MINMAX)
    subimageF1 = cv2.normalize(get_gabor(subimageF1)[...,2], subimageF1, 0, 1,\
        cv2.NORM_MINMAX)

    #dif_img = np.ones_like(subimageF1)

    for corner in corners_s:
        xmin = int(corner[0]) - deltas[0]
        xmax = int(corner[0]) + deltas[1] + 1
        ymin = int(corner[1]) - deltas[2]
        ymax = int(corner[1]) + deltas[3] + 1

        if xmin >= 0 and ymin >= 0 and xmax <= subimageF2.shape[1] and\
                                       ymax <= subimageF2.shape[0]:
            weights[i] = np.sum((subimageF2[ymin:ymax, xmin:xmax] * subimageF1),\
                dtype=np.float32) / np.sqrt(\
                np.sum(subimageF2[ymin:ymax,xmin:xmax]**2, dtype=np.float32)*\
                    np.sum(subimageF1**2, dtype=np.float32))
        i+=1

    corner = corners[np.argmax(weights),0]
    return corner + offset


def getPFromRectangleACorespondingP(img1, img2, point1, point2):
    """Wrapper for getPointFromRectangle and getCorespondingPoint.

    Arguments
    ---------
    img1 : ndarray
        Image in which point is searched.
    img2 : ndarray
        Image in which corresponding point is searched.
    point1 : tuple
        First corner of rectangle in which point is searched.
    point2 : tuple
        Second corner of rectangle in which point is searched.
    
    Returns
    -------
    returnPoint1 : tuple
        Found point.
    returnPoint2 :  tuple
        Corresponding point.

    """
    returnPoint1 = getPointFromRectangle(img1, point1, point2)
    returnPoint2 = getCorespondingPoint(img1, img2, returnPoint1)
    returnPoint2 = (returnPoint2[0], returnPoint2[1])
    return returnPoint1, returnPoint2


def getPointFromPoint(img, point):
    """Returns most fitting point near a given point in an image.

    Parameters
    ----------
    img : ndarray
        Image in which point is searched.
    point : tuple
        Point near which the best point/corner is searched.
    
    Returns
    -------
    point : tuple
        Best point.

    """
    rect_size_half = int(((img.shape[0] + img.shape[1]) / 2 ) * 0.01)
    
    point1 = (max(0, point[0] - rect_size_half),\
              max(0, point[1] - rect_size_half))
    point2 = (min(img.shape[1], point[0] + rect_size_half),\
              min(img.shape[0], point[1] + rect_size_half))

    return getPointFromRectangle(img, point1, point2)


