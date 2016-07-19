import cv2
import numpy as np

# scale factor for subimage size on which template matching is done
SUBIMAGE_SIZE_SCALE = 0.159  
# scale factor for template size
TEMPLATE_SIZE_SCALE = 0.053
# if rectangle is smaller compute center point of two points
MIN_RECT_SIZE = 5

# TODO delete for build
def myFilledCircle(img, center):
	radiusSize = 0.005
	thickness = -1
	lineType = 8
	shape = img.shape
	radius = int((shape[0] + shape[1]) * radiusSize)
	cv2.circle (img, center, radius + 2, (255,255,255), thickness, lineType)
	cv2.circle (img, center, radius, (0,0,0), thickness, lineType)

class sac:
	def __init__(self, img1, img2):
		self.img1 = img1
		self.img2 = img2
		# size of local subimage in which dense sampling is done
		self.subimageSize = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5 * SUBIMAGE_SIZE_SCALE)
		# diameter of the meaningfull keypoint neighborhood 
		self.templateSize = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5  * TEMPLATE_SIZE_SCALE)

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

		#TODO remove for build
		'''	
		corners = cv2.normalize(corners, corners, 0, 255, cv2.NORM_MINMAX)
		cv2.imshow("corners", np.uint8(corners))
		cv2.imshow("gaus", gausMatrixNormalized)
		cv2.imshow("subimage", np.uint8(subimageF*255))
		''' 

		return returnPoint

	def getCorespondingPoint(self, point, imageSelect):
		"""Search for coresponding point on second image given a point in first image using template matching."""

		# select image on which user draw
		if imageSelect:
			img1 = self.img1
			img2 = self.img2
		else:
			img1 = self.img2
			img2 = self.img1

		assert 0 <= point[1] < img1.shape[0], "Point outside image 1. Have both images the same size?"
		assert 0 <= point[0] < img1.shape[1], "Point outside image 1. Have both images the same size?"
		assert 0 <= point[1] < img2.shape[0], "Point outside image 2. Have both images the same size?"
		assert 0 <= point[0] < img2.shape[1], "Point outside image 2. Have both images the same size?"

		# get template from img1 in which user draw
		templateSizeHalf = int(self.templateSize/2)
		x1 = max(point[0] - templateSizeHalf, 0)
		x2 = min(point[0] + templateSizeHalf, img1.shape[1] - 1)
		y1 = max(point[1] - templateSizeHalf, 0)
		y2 = min(point[1] + templateSizeHalf, img1.shape[0] - 1)
		subimage1 = np.copy(img1[y1:y2, x1:x2])

		# create subimage from img2 in which template is searched
		subimageSizeHalf = int(self.subimageSize/2)
		x1 = max(point[0] - subimageSizeHalf, 0)
		x2 = min(point[0] + subimageSizeHalf, img2.shape[1] - 1)
		y1 = max(point[1] - subimageSizeHalf, 0)
		y2 = min(point[1] + subimageSizeHalf, img2.shape[0] - 1)
		subimage2 = np.copy(img2[y1:y2, x1:x2])

		# preprocess both subimages
		subimage1F = np.float32(subimage1)
		subimage1F = cv2.cvtColor(subimage1F, cv2.COLOR_BGR2GRAY)
		subimage1F = cv2.normalize(subimage1F, subimage1F, 0, 1, cv2.NORM_MINMAX)
	 	subimage1F = cv2.GaussianBlur(subimage1F, (5,5), 0)	
		subimage1X = cv2.Scharr(subimage1F, ddepth = -1, dx = 1, dy = 0)
		subimage1Y = cv2.Scharr(subimage1F, ddepth = -1, dx = 0, dy = 1)
		subimage1F = subimage1X + subimage1Y
		subimage1F = cv2.normalize(subimage1F, subimage1F, 0, 1, cv2.NORM_MINMAX)

		subimage2F = np.float32(subimage2)
		subimage2F = cv2.cvtColor(subimage2F, cv2.COLOR_BGR2GRAY)
		subimage2F = cv2.normalize(subimage2F, subimage2F, 0, 1, cv2.NORM_MINMAX)
	 	subimage2F = cv2.GaussianBlur(subimage2F, (5,5), 0)	
		subimage2X = cv2.Scharr(subimage2F, ddepth = -1, dx = 1, dy = 0)
		subimage2Y = cv2.Scharr(subimage2F, ddepth = -1, dx = 0, dy = 1)
		subimage2F = subimage2X + subimage2Y
		subimage2F = cv2.normalize(subimage2F, subimage2F, 0, 1, cv2.NORM_MINMAX)

		# template matching
		# norms are missing in cv2 python wrapper
		CV_TM_SQDIFF = 0
		CV_TM_SQDIFF_NORMED = 1
		CV_TM_CCORR = 2
		CV_TM_CCORR_NORMED = 3
		CV_TM_CCOEFF = 4
		CV_TM_CCOEFF_NORMED = 5
		templateResult = cv2.matchTemplate(subimage2F, subimage1F, CV_TM_CCOEFF_NORMED)
		templateResult1 = cv2.normalize(templateResult, templateResult, 0, 1, cv2.NORM_MINMAX)

		minVal, maxVal, minLoc , maxLoc = cv2.minMaxLoc(templateResult1)

		point2 = (maxLoc[0] + templateSizeHalf, maxLoc[1] + templateSizeHalf)

		returnPoint = (x1 + int(point2[0]), y1 + int(point2[1]))
		return returnPoint

	def getPFromRectangleACorespondingP(self, point1, point2, imageSelect):
		"""Wrapper for getPointFromRectangle and getCorespondingPoint.
		imageSelect: True if user draw rect on image 1, False if user draw on image 2.
		"""
		returnPoint1 = self.getPointFromRectangle(point1, point2, imageSelect)
		returnPoint2 = self.getCorespondingPoint(returnPoint1, imageSelect)
		return returnPoint1, returnPoint2
