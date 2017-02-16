import cv2
import numpy as np
from image_helpers import lce
from image_helpers import draw_circle
from image_gabor import get_gabor

# scale factor for subimage size on which template matching is done
SUBIMAGE_SIZE_SCALE = 0.3 #0.159  
# scale factor for template size
TEMPLATE_SIZE_SCALE = 0.07
# minimum size of user drawn rectangle to start point search else return middle point
MIN_RECT_SIZE = 5


def getPointFromRectangle(img1, point1, point2):
    """Computes point of interest in a subimage which is defined by to given points."""

    assert 0 <= point1[1] < img1.shape[0], "Point1 outside image"
    assert 0 <= point1[0] < img1.shape[1], "Point1 outside image"
    assert 0 <= point2[1] < img1.shape[0], "Point2 outside image"
    assert 0 <= point2[0] < img1.shape[1], "Point2 outside image"
    
    # if rectangle is to small return middlepoint of the two given points, assuming user 
    # wanted to select a single point and not draw rectangle
    if abs(point1[0] - point2[0]) < MIN_RECT_SIZE or abs(point1[1] - point2[1]) < MIN_RECT_SIZE:
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
    corners = cv2.cornerHarris( subimageF, blockSize, apertureSize, k, borderType=cv2.BORDER_DEFAULT )

    # Assume that user wants to mark point in middle of rectangle, hence weight cornes using gaussian
    rows, cols = corners.shape
    gausCols = cv2.getGaussianKernel(cols, -1)
    gausRows = cv2.getGaussianKernel(rows, -1)
    gausMatrix = gausRows*gausCols.T
    gausMatrixNormalized = gausMatrix/gausMatrix.max()
    corners = corners * gausMatrixNormalized
    
    # get sharpest corners
    i, j = np.where(corners == corners.max());

    # get index of corner in middle of sharpest corner array, most often there is only one entry 
    index = int(i.shape[0]/2)

    #add the start position of rectangle as offset
    returnPoint = (j[index] + min(point1[0], point2[0]), i[index] + min(point1[1], point2[1]))

    return returnPoint


def get_and_pre_patch(img, point, size_half):
    x1 = max(point[0] - size_half, 0)
    x2 = min(point[0] + size_half, img.shape[1])
    y1 = max(point[1] - size_half, 0)
    y2 = min(point[1] + size_half, img.shape[0])
    delta_x_min = point[0] - x1
    delta_x_max = x2 - point[0]
    delta_y_min = point[1] - y1
    delta_y_max = y2 - point[1]
    subimage = cv2.cvtColor(np.float32(img[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV).copy()
    #subimage = cv2.cvtColor(np.float32(img[y1:y2, x1:x2]), cv2.COLOR_BGR2HSV)[...,2].copy()
#    subimage = cv2.normalize(subimage, subimage, 0, 255 , cv2.NORM_MINMAX)
#    lutable = np.uint8(np.arange(8).repeat(32)*36.4286)
#    subimage = np.float32(cv2.LUT(np.uint8(subimage), lutable))
    subimage = cv2.normalize(subimage, subimage, 0.0, 1.0 , cv2.NORM_MINMAX)
    #return cv2.GaussianBlur(subimage, (5,5), 0), np.array((x1, y1)), np.array([delta_x_min, delta_x_max, delta_y_min, delta_y_max], dtype=np.float32)   
    return subimage, np.array((x1, y1)), np.array([delta_x_min, delta_x_max, delta_y_min, delta_y_max], dtype=np.float32)   

def getCorespondingPoint(img1, img2, point, template_size_s=101):
    """Search for coresponding point on second image given a point in first image using template matching."""

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
    corners = cv2.goodFeaturesToTrack(np.uint8(subimageF2[...,2]*255), 2000, qualityLevel=0.01, minDistance=10)

    # Scale patches, lines and points
    sf = template_size_s/np.float32(np.max(subimageF1.shape[0:2]))
    #subimageF1 = cv2.GaussianBlur(cv2.resize(subimageF1, (0,0), fx=sf, fy=sf), (5,5), 0)
    #subimageF1 = cv2.GaussianBlur(cv2.resize(subimageF1, (0,0), fx=sf, fy=sf), (5,5), 0)
    subimageF2 = cv2.resize(subimageF2, (0,0), fx=sf, fy=sf)
    subimageF1 = cv2.resize(subimageF1, (0,0), fx=sf, fy=sf)
    corners_s = [corner[0] * sf for corner in corners]
    deltas = np.int_(deltas * sf)

    templateSizeHalf_s = int(templateSizeHalf * sf)  
#    weights = np.zeros((corners.shape[0]), dtype=np.float32)
    weights = np.full((corners.shape[0]), np.inf, dtype=np.float32)
    patch = np.empty(subimageF1.shape)
    i = 0
    min_w = np.inf
    max_w = 0

    #cv2.imshow('i2', np.uint8(subimageF2*255))
    #cv2.imshow('i1', np.uint8(subimageF1*255))
    #subimageF2 = cv2.normalize(get_gabor(subimageF2)[...,2], subimageF2, 0, 1, cv2.NORM_MINMAX)
    #subimageF1 = cv2.normalize(get_gabor(subimageF1)[...,2], subimageF1, 0, 1, cv2.NORM_MINMAX)

    #dif_img = np.ones_like(subimageF1)
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #hog= cv2.HOGDescriptor()


    i1 = hog.compute(np.uint8(subimageF1[...,2]*255), (8,8), (8,8))
#    cv2.imshow('f2', np.uint8(subimageF2*255))
#    cv2.imshow('f1', np.uint8(subimageF1*255))
#    cv2.waitKey()

    for corner in corners_s:
        xmin = int(corner[0]) - deltas[0]
        xmax = int(corner[0]) + deltas[1] + 1
        ymin = int(corner[1]) - deltas[2]
        ymax = int(corner[1]) + deltas[3] + 1

        if xmin >= 0 and ymin >= 0 and xmax <= subimageF2.shape[1] and\
                                       ymax <= subimageF2.shape[0]:

            i2 = hog.compute(np.uint8(subimageF2[ymin:ymax, xmin:xmax,2]*255), (8,8), (8,8))
            weights[i] = np.sum((i1 - i2)**2, dtype=np.float32)


            # sqdiff_normed minimum is best
            #weights[i] = np.sum((subimageF2[ymin:ymax, xmin:xmax] - subimageF1)**2, dtype=np.float32)/\
            #   np.sqrt(np.sum(subimageF2[ymin:ymax,xmin:xmax]**2, dtype=np.float32) * np.sum(subimageF1**2, dtype=np.float32))

            #crosscorelation_normed maximum is best
#            weights[i] = np.sum((subimageF2[ymin:ymax, xmin:xmax] * subimageF1), dtype=np.float32)/\
#                np.sqrt(np.sum(subimageF2[ymin:ymax,xmin:xmax]**2, dtype=np.float32) * np.sum(subimageF1**2, dtype=np.float32))

            # hamming distance
#            dif_img[(subimageF2[ymin:ymax, xmin:xmax]-subimageF1)**2<.01] = 0
#            cv2.imshow('di', np.uint8(dif_img*255))
#            weights[i] = np.sum(dif_img, dtype=np.float32)
#            dif_img[:] = 1

#            if weights[i]  > max_w:
#                print weights[i]
#                max_w = weights[i]
#                cv2.imshow('patch',  np.uint8(subimageF1*255))
#                draw_circle(subimageF2, tuple(np.int_(corner)), (1))
#                cv2.imshow('patch2',  np.uint8(subimageF2[ymin:ymax, xmin:xmax]*255))
#                k = cv2.waitKey(0)
#                if k == 27: break

#        idx_min = 0 if xmin >= 0 else abs(xmin)
#        idy_min = 0 if ymin >= 0 else abs(ymin)
#        idx_max = subimageF1.shape[0] if xmax < subimageF2.shape[0] else\
#                  subimageF1.shape[0] - (xmax-subimageF2.shape[0])
#        idy_max = subimageF1.shape[1] if ymax < subimageF2.shape[1] else\
#                  subimageF1.shape[1] - (ymax-subimageF2.shape[1])
#
#        num_elem = (idx_max - idx_min) * (idy_max - idy_min)
#
#        weights[i] = np.sum((subimageF2\
#                                [max(xmin,0):min(xmax,subimageF2.shape[0]),\
#                                 max(ymin,0):min(ymax,subimageF2.shape[1])] -\
#                             subimageF1[idx_min:idx_max,idy_min:idy_max])**2)/\
#                     num_elem
#        print weights[i]
        i+=1

    corner = corners[np.argmin(weights),0]
#    draw_circle(subimageF2, tuple(np.int_(corners_s[np.argmin(weights)])), (1))
#    cv2.imshow('corners', np.uint8(cv2.normalize(subimageF2, subimageF2, 0, 255, cv2.NORM_MINMAX)))


#    return offset
    return corner + offset


def getPFromRectangleACorespondingP(img1, img2, point1, point2):
    """Wrapper for getPointFromRectangle and getCorespondingPoint.
    imageSelect: True if user draw rect on image 1, False if user draw on image 2.
    """
    returnPoint1 = getPointFromRectangle(img1, point1, point2)
    returnPoint2 = getCorespondingPoint(img1, img2, returnPoint1)
    #returnPoint2 = getPointFromRectangle(img2, (returnPoint1[0]-20,returnPoint1[1]-20), (returnPoint1[0]+20, returnPoint1[1]+20))
    return returnPoint1, returnPoint2

def getPointFromPoint(img, point):
    rect_size_half = int(((img.shape[0] + img.shape[1]) / 2 ) * 0.01)
    
    point1 = (max(0, point[0] - rect_size_half), max(0, point[1] - rect_size_half))
    point2 = (min(img.shape[1], point[0] + rect_size_half), min(img.shape[0], point[1] + rect_size_half))

    return getPointFromRectangle(img, point1, point2)


