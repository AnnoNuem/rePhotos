import cv2
import numpy as np

FLANN_INDEX_KDTREE = 1
# scale factor for rectangle in which dense sampling is done
RECTANGLE_SIZE_SCALE = 0.159  
# scale factor for area taken into accout for descriptor computation
KEYPOINT_SIZE_SCALE = 0.053
# scale factor for grid line distance
GRID_SCALE = 0.034
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
		self.detector = cv2.xfeatures2d.SURF_create(800, extended = True, upright = True)
		self.norm = cv2.NORM_L2
		self.flann_params = dict(algorithm = 1 , trees = 5)
		self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})  # bug : need to pass empty dict (#1329)
		# size of local subimage in which dense sampling is done
		self.rectangle_size = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5 * RECTANGLE_SIZE_SCALE)
		# diameter of the meaningfull keypoint neighborhood 
		self.keypoint_size = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5  * KEYPOINT_SIZE_SCALE)
		# distance between grid lines in pixels for dense sampling
		self.grid_size = int(self.rectangle_size * GRID_SCALE)


	def getPointFromRectangle(self, point1, point2, image_select):
		""" computes point of interest in a subimage which is defined by to given points"""

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

		# get index of corner in middle of subimage
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
		""" search for coresponding point on second image given a point in first image using dense sampling"""

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

		# get subimage from img1 to compute descriptor of user selected point in img1
		# subimage has to have size of region used for descriptor computation
		rectangle_size_half_1 = int(self.keypoint_size/2)
		x1 = max(point[0] - rectangle_size_half_1, 0)
		x2 = min(point[0] + rectangle_size_half_1, img1.shape[1] - 1)
		y1 = max(point[1] - rectangle_size_half_1, 0)
		y2 = min(point[1] + rectangle_size_half_1, img1.shape[0] - 1)
		subimage1 = np.copy(img1[y1:y2, x1:x2])

		# create subimage from img2 in which the coresponding point is searched
		rectangle_size_half = int(self.rectangle_size/2)
		x1 = max(point[0] - rectangle_size_half, 0)
		x2 = min(point[0] + rectangle_size_half, img2.shape[1] - 1)
		y1 = max(point[1] - rectangle_size_half, 0)
		y2 = min(point[1] + rectangle_size_half, img2.shape[0] - 1)
		subimage2 = np.copy(img2[y1:y2, x1:x2])

		# preprocess both subimages
		subimage1 = cv2.cvtColor(subimage1, cv2.COLOR_BGR2GRAY)
	 	subimage1 = cv2.GaussianBlur(subimage1, (5,5), 0)	
		subimage2 = cv2.cvtColor(subimage2, cv2.COLOR_BGR2GRAY)
	 	subimage2 = cv2.GaussianBlur(subimage2, (5,5), 0)	
		
		# create keypoint and compute descriptors for point in subimg1
		keypoints_img1 = [cv2.KeyPoint(rectangle_size_half_1, rectangle_size_half_1, self.keypoint_size)]
		k, descriptors_img1 = self.detector.compute(subimage1, keypoints_img1)

		# generate keypoints for img2
		keypoints_img2 = [cv2.KeyPoint(x, y, self.keypoint_size) for x in range(0, subimage2.shape[1], self.grid_size) for y in range(0, subimage2.shape[0], self.grid_size)]
		
		# compute descriptors for subimage of img2
		k2, descriptors_img2 = self.detector.compute(subimage2, keypoints_img2)

		# flann
		match = self.matcher.match(descriptors_img1, trainDescriptors = descriptors_img2)
		point2 = np.float32(keypoints_img2[match[0].trainIdx].pt)
		
		# TODO delete for build
		myFilledCircle(subimage2, (int(point2[0]), int(point2[1])))
		cv2.imshow("subimage1", subimage1)
		cv2.imshow("subimage2", subimage2)

		return_point = (x1 + int(point2[0]), y1 + int(point2[1]))
		return return_point

	def getPFromRectangleACorespondingP(self, point1, point2, image_select):
		""" 
		wrapper for getPointFromRectangle and getCorespondingPoint 
		image_select: True if user draw rect on image 1, False if user draw on image 2
		"""
		return_point1 = self.getPointFromRectangle(point1, point2, image_select)
		return_point2 = self.getCorespondingPoint(return_point1, image_select)
		return return_point1, return_point2
