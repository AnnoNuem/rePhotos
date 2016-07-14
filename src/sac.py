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
	radius_size = 0.005
	thickness = -1
	lineType = 8
	shape = img.shape
	radius = int((shape[0] + shape[1]) * radius_size)
	cv2.circle (img, center, radius + 2, (255,255,255), thickness, lineType)
	cv2.circle (img, center, radius, (0,0,0), thickness, lineType)

class sac:
	def __init__(self, img1, img2):
		self.img1 = img1
		self.img2 = img2
		# size of local subimage in which dense sampling is done
		self.subimage_size = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5 * SUBIMAGE_SIZE_SCALE)
		# diameter of the meaningfull keypoint neighborhood 
		self.template_size = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5  * TEMPLATE_SIZE_SCALE)

	def getPointFromRectangle(self, point1, point2, image_select):
		"""Computes point of interest in a subimage which is defined by to given points."""

		# select image on which user draw
		if image_select:
			img1 = self.img1
		else:
			img1 = self.img2

		assert 0 < point1[1] < img1.shape[0], "Point1 outside image"
		assert 0 < point1[0] < img1.shape[1], "Point1 outside image"
		assert 0 < point2[1] < img1.shape[0], "Point2 outside image"
		assert 0 < point2[0] < img1.shape[1], "Point2 outside image"
		
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
		blockSize = 2
		apertureSize = 3
		k = 0.04
		# Detecting corners
		corners = cv2.cornerHarris( subimage_f, blockSize, apertureSize, k, cv2.BORDER_DEFAULT )

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

		#TODO remove for build
		'''	
		corners = cv2.normalize(corners, corners, 0, 255, cv2.NORM_MINMAX)
		cv2.imshow("corners", np.uint8(corners))
		cv2.imshow("gaus", gaus_matrix_normalized)
		cv2.imshow("subimage", np.uint8(subimage_f*255))
		''' 

		return return_point

	def getCorespondingPoint(self, point, image_select):
		"""Search for coresponding point on second image given a point in first image using template matching."""

		# select image on which user draw
		if image_select:
			img1 = self.img1
			img2 = self.img2
		else:
			img1 = self.img2
			img2 = self.img1

		assert 0 < point[1] < img1.shape[0], "Point outside image 1. Have both images the same size?"
		assert 0 < point[0] < img1.shape[1], "Point outside image 1. Have both images the same size?"
		assert 0 < point[1] < img2.shape[0], "Point outside image 2. Have both images the same size?"
		assert 0 < point[0] < img2.shape[1], "Point outside image 2. Have both images the same size?"

		# get template from img1 in which user draw
		template_size_half = int(self.template_size/2)
		x1 = max(point[0] - template_size_half, 0)
		x2 = min(point[0] + template_size_half, img1.shape[1] - 1)
		y1 = max(point[1] - template_size_half, 0)
		y2 = min(point[1] + template_size_half, img1.shape[0] - 1)
		subimage1 = np.copy(img1[y1:y2, x1:x2])

		# create subimage from img2 in which template is searched
		subimage_size_half = int(self.subimage_size/2)
		x1 = max(point[0] - subimage_size_half, 0)
		x2 = min(point[0] + subimage_size_half, img2.shape[1] - 1)
		y1 = max(point[1] - subimage_size_half, 0)
		y2 = min(point[1] + subimage_size_half, img2.shape[0] - 1)
		subimage2 = np.copy(img2[y1:y2, x1:x2])

		# preprocess both subimages
		subimage1_f = np.float32(subimage1)
		subimage1_f = cv2.cvtColor(subimage1_f, cv2.COLOR_BGR2GRAY)
		subimage1_f = cv2.normalize(subimage1_f, subimage1_f, 0, 1, cv2.NORM_MINMAX)
	 	subimage1_f = cv2.GaussianBlur(subimage1_f, (5,5), 0)	
		subimage1_x = cv2.Scharr(subimage1_f, ddepth = -1, dx = 1, dy = 0)
		subimage1_y = cv2.Scharr(subimage1_f, ddepth = -1, dx = 0, dy = 1)
		subimage1_f = subimage1_x + subimage1_y
		subimage1_f = cv2.normalize(subimage1_f, subimage1_f, 0, 1, cv2.NORM_MINMAX)

		subimage2_f = np.float32(subimage2)
		subimage2_f = cv2.cvtColor(subimage2_f, cv2.COLOR_BGR2GRAY)
		subimage2_f = cv2.normalize(subimage2_f, subimage2_f, 0, 1, cv2.NORM_MINMAX)
	 	subimage2_f = cv2.GaussianBlur(subimage2_f, (5,5), 0)	
		subimage2_x = cv2.Scharr(subimage2_f, ddepth = -1, dx = 1, dy = 0)
		subimage2_y = cv2.Scharr(subimage2_f, ddepth = -1, dx = 0, dy = 1)
		subimage2_f = subimage2_x + subimage2_y
		subimage2_f = cv2.normalize(subimage2_f, subimage2_f, 0, 1, cv2.NORM_MINMAX)

		# template matching
		# norms are missing in cv2 python wrapper
		CV_TM_SQDIFF = 0
		CV_TM_SQDIFF_NORMED = 1
		CV_TM_CCORR = 2
		CV_TM_CCORR_NORMED = 3
		CV_TM_CCOEFF = 4
		CV_TM_CCOEFF_NORMED = 5
		template_result = cv2.matchTemplate(subimage2_f, subimage1_f, CV_TM_CCOEFF_NORMED)
		template_result_1 = cv2.normalize(template_result, template_result, 0, 1, cv2.NORM_MINMAX)
		minVal, maxVal, minLoc , maxLoc = cv2.minMaxLoc(template_result_1)

		point2 = (maxLoc[0] + template_size_half, maxLoc[1] + template_size_half)

		# TODO delete for build
		myFilledCircle(subimage2_f, (int(point2[0]), int(point2[1])))
		cv2.imshow("subimage1", np.uint8(subimage1_f*255))
		cv2.imshow("subimage2", np.uint8(subimage2_f*255))
		cv2.imshow("template result", np.uint8(template_result_1 * 255))

		return_point = (x1 + int(point2[0]), y1 + int(point2[1]))
		return return_point

	def getPFromRectangleACorespondingP(self, point1, point2, image_select):
		"""Wrapper for getPointFromRectangle and getCorespondingPoint.
		image_select: True if user draw rect on image 1, False if user draw on image 2
		"""
		return_point1 = self.getPointFromRectangle(point1, point2, image_select)
		return_point2 = self.getCorespondingPoint(return_point1, image_select)
		return return_point1, return_point2
