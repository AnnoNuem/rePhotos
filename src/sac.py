import cv2
import numpy as np

def getPointFromRectangle(img1, point1, point2):
	"""
	Computes point of interest in a subimage which is defined by to given points.
	:param img1: image in which point is searched
	:param point1: corner of user drawn rectangle
	:param point2: opposite corner of user drawn rectangle
	:return: point of interest in rectangle"""


	assert 0 <= point1[1] < img1.shape[0], "Point1 outside image"
	assert 0 <= point1[0] < img1.shape[1], "Point1 outside image"
	assert 0 <= point2[1] < img1.shape[0], "Point2 outside image"
	assert 0 <= point2[0] < img1.shape[1], "Point2 outside image"

	assert point1[0] != point2[0], "X cordinates of rectangle corners are equal -> no rectangle"
	assert point1[1] != point2[1], "Y cordinates of rectangle corners are equal -> no rectangle"
	
	subimage = np.copy(img1[min(point1[1],point2[1]):max(point1[1],point2[1]), 
										  min(point1[0], point2[0]):max(point1[0],point2[0])])
	subimageGray = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)
	subimageF = np.float32(subimageGray)
	subimageF = cv2.normalize(subimageF, subimageF, 0, 1, cv2.NORM_MINMAX)
	subimageF = cv2.GaussianBlur(subimageF, (5,5), 0)	
	
	# Detector parameters
	blockSize = 2
	apertureSize = 3
	k = 0.04
	# Detecting corners
	corners = cv2.cornerHarris( subimageF, blockSize, apertureSize, k, cv2.BORDER_DEFAULT )

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

