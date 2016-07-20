import numpy as np
import cv2

def transformPoints(points, transformMatrix):
	"""
	Transforms a list of points given a transformation matrix.
	:param points: the list of points given as touples
	:param transformMatrix: the transformation matrix
	"""
	# TODO not working properly
	pointsTransformed = []
	for point in points:
		pArray = np.array([point[0], point[1], 1])
		pArrayTransformed = transformMatrix.dot(pArray)
		pointsTransformed.append((int(pArrayTransformed[0]), int(pArrayTransformed[1])))
	return pointsTransformed

def calibrate(img1, img2, pointsImg1, pointsImg2, alpha = 0.5):
	"""
	Aligns the two images with the best matching perspective transform given the two point lists.
	Points in pointlists are transformed as well.
	:param img1: Image 1
	:param img2: Image 2
	:param pointsImg1: marked points in image 1
	:param pointsImg2: coresponding points in image 2
	:param alpha: 0 = align img2 to img1, 1 = align img1 to img2, 0.5 align img1 and img2 to mean
		and points acordingly.
	"""

	assert 0 <= alpha <= 1, "Alpha not between 0 and 1."
	assert len(pointsImg1) == len(pointsImg2), "Point lists of unequal length"
	assert len(pointsImg1) > 3, "Not enough points to find homography"

	if alpha == 0:
		transformMatrix, _ = cv2.findHomography(np.vstack(pointsImg2).astype(float), np.vstack(pointsImg1).astype(float), 0)
		img2 = cv2.warpPerspective(img2, transformMatrix, (img1.shape[1], img1.shape[0]))
		transformPoints(pointsImg2, transformMatrix)	
	elif alpha == 1:
		transformMatrix, _ = cv2.findHomography(np.vstack(pointsImg1).astype(float), np.vstack(pointsImg2).astype(float), 0)
		img1 = cv2.warpPerspective(img1, transformMatrix, (img2.shape[1], img2.shape[0]))
		pointsImg1 =  transformPoints(pointsImg1, transformMatrix)	
	else:
		pointsDest = []
		alphaM1 = 1 - alpha
		xMaxDest = int( alphaM1 * img1.shape[1] + alpha * img2.shape[1] / 2)
		yMaxDest = int( alphaM1 * img1.shape[0] + alpha * img2.shape[0] / 2)
		i = 0
		for pointImg1 in pointsImg1:
			pointsDest.append((int(alphaM1 * pointImg1[0] + alpha * (pointsImg2[i])[0] / 2),\
					  				 int(alphaM1 * pointImg1[1] + alpha * (pointsImg2[i])[1] / 2))) 
			i += 1
		transformMatrix1, _ = cv2.findHomography(np.vstack(pointsImg1).astype(float), np.vstack(pointsDest).astype(float), 0)
		transformMatrix2, _ = cv2.findHomography(np.vstack(pointsImg2).astype(float), np.vstack(pointsDest).astype(float), 0)
		img1 = cv2.warpPerspective(img1, transformMatrix1, (xMaxDest, yMaxDest))
		img2 = cv2.warpPerspective(img2, transformMatrix2, (xMaxDest, yMaxDest))
		pointsImg1 = pointsImg2 = pointsDest


	# TODO remove for build
	result = img1 * 0.5 + img2 * 0.5
	result = cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
	cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
	#cv2.namedWindow("img1 cali", cv2.WINDOW_KEEPRATIO)
	#cv2.namedWindow("img2 cali", cv2.WINDOW_KEEPRATIO)
	cv2.imshow("result", np.uint8(result) )
	#cv2.imshow("img1 cali", img1)
	#cv2.imshow("img2 cali", img2)
	#cv2.resizeWindow("img1 cali", 640, 480)
	#cv2.resizeWindow("img2 cali", 640, 480)
	cv2.resizeWindow("result", 640, 480)

	return img1, img2, pointsImg1, pointsImg2