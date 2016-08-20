import numpy as np
from numpy.linalg import inv
import cv2

def postProcess(img1, img2):
    """
    Crops images.
    """
    pass

def transformPoints(points, transformMatrix):
    """
    Transforms a list of points given a transformation matrix.
    :param points: the list of points given as touples
    :param transformMatrix: the transformation matrix
    """
    pointArray = np.float32(points)
    pointArray = np.int32(cv2.perspectiveTransform(pointArray.reshape(1,-1,2), transformMatrix).reshape(-1, 2)).tolist()

    pointsTransformed = []
    for point in pointArray:
        pointsTransformed.append(tuple(point))

    return pointsTransformed

def calibrate(img1, img2, pointsImg1, pointsImg2, alpha = None):
    """
    Aligns the two images with the best matching perspective transform given the two point lists.
    Points in pointlists are transformed as well.
    :param img1: Image 1
    :param img2: Image 2
    :param pointsImg1: marked points in image 1
    :param pointsImg2: coresponding points in image 2
    :param alpha: 0 = align img2 to img1, 1 = align img1 to img2, 0.5 align img1 and img2 to mean
        and points acordingly. If alpha is None the smaller image is transformed to bigger one
    """

    assert len(pointsImg1) == len(pointsImg2), "Point lists of unequal length"
    assert len(pointsImg1) > 3, "Not enough points to find homography"

    if (alpha == None):
        if (img1.shape[0] * img1.shape[1]) < (img2.shape[0] * img2.shape[1]):
            alpha = 1
        else:
            alpha = 0

    assert 0 <= alpha <= 1, "Alpha not between 0 and 1."

    # Keep image 1
    if alpha == 0:
        transformMatrix, _ = cv2.findHomography(np.vstack(pointsImg2).astype(float), np.vstack(pointsImg1).astype(float), 0)
        img2 = cv2.warpPerspective(img2, transformMatrix, (img1.shape[1], img1.shape[0]))
        pointsImg2 = transformPoints(pointsImg2, transformMatrix)   
    # Keep image 2
    elif alpha == 1:
        transformMatrix, _ = cv2.findHomography(np.vstack(pointsImg1).astype(float), np.vstack(pointsImg2).astype(float), 0)
        y, x, _ = img1.shape
        cornersImg1= [(0,0), (0,y), (x,y), (x,0)]
        cornersImg1 = transformPoints(cornersImg1, inv(transformMatrix))
        print cornersImg1
        # TODO corner berechnung und cropping funktioniert so garnicht
        img1 = cv2.warpPerspective(img1, transformMatrix, (img2.shape[1], img2.shape[0]))
        img1 = img1[max((cornersImg1[0])[0],0): min((cornersImg1[3])[0], img2.shape[1]),\
            max((cornersImg1[0])[0], 0): min((cornersImg1[3])[1], img2.shape[0]),:]
        pointsImg1 =  transformPoints(pointsImg1, transformMatrix)  
    # Transform  both image with respect to alpha
    else:
        pointsDest = []
        alphaM1 = 1 - alpha
        xMaxDest = int(( alphaM1 * img1.shape[1] + alpha * img2.shape[1]) / 2)
        yMaxDest = int(( alphaM1 * img1.shape[0] + alpha * img2.shape[0]) / 2)
        i = 0
        for pointImg1 in pointsImg1:
            pointsDest.append((int(alphaM1 * pointImg1[0] + alpha * (pointsImg2[i])[0] / 2),\
                                     int(alphaM1 * pointImg1[1] + alpha * (pointsImg2[i])[1] / 2))) 
            i += 1
        transformMatrix1, _ = cv2.findHomography(np.vstack(pointsImg1).astype(float), np.vstack(pointsDest).astype(float), 0)
        transformMatrix2, _ = cv2.findHomography(np.vstack(pointsImg2).astype(float), np.vstack(pointsDest).astype(float), 0)
        img1 = cv2.warpPerspective(img1, transformMatrix1, (xMaxDest, yMaxDest))
        img2 = cv2.warpPerspective(img2, transformMatrix2, (xMaxDest, yMaxDest))
        pointsImg1  =  pointsDest[:]
        pointsImg2  =  pointsDest[:]



    # TODO remove for build
    #result = img1 * 0.5 + img2 * 0.5
    #result = cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
    #cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("result", np.uint8(result) )
    #cv2.resizeWindow("result", 640, 480)

    return img1, img2, pointsImg1, pointsImg2
