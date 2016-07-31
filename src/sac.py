import cv2
import numpy as np

# if rectangle is smaller compute center point of two points
MIN_RECT_SIZE = 5

class sac:
	def __init__(self, img1, img2):
		self.img1 = img1
		self.img2 = img2


	def getPointFromRectangle(self, point1, point2, imageSelect):
		"""Computes point of interest in a subimage which is defined by to given points."""

		# select image on which user draw
		if imageSelect:
			img1 = self.img1
		else:
			img1 = self.img2

		assert 0 <= point1[1] < img1.shape[0], "Point1 outside image"
		assert 0 <= point1[0] < img1.shape[1], "Point1 outside image"
		assert 0 <= point2[1] < img1.shape[0], "Point2 outside image"
		assert 0 <= point2[0] < img1.shape[1], "Point2 outside image"
		
		# if rectangle is to small return middlepoint of the two given points, assuming user 
		# wanted to select a single point and not draw rectangle
		if abs(point1[0] - point2[0]) < MIN_RECT_SIZE or abs(point1[1] - point2[1]) < MIN_RECT_SIZE:
			return (int((point1[0]+point2[0])/2), int((point1[1]+point2[1])/2))

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

