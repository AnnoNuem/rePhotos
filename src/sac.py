import copy
import cv2
import numpy as np

# scale factor for subimage size on which template matching is done
SUBIMAGE2_SIZE_SCALE = 0.159  
# scale factor for template size
TEMPLATE_SIZE_SCALE = 0.053
# if rectangle is smaller compute center point of two points
MIN_RECT_SIZE = 5
# scale factor for subimage in which point near user selected point is searched
SUBIMAGE1_SIZE_SCALE = 0.01

# TODO delete for build
def myFilledCircle(img, center):
	radius_size = 0.005
	thickness = -1
	lineType = 8
	shape = img.shape
	radius = int((shape[0] + shape[1]) * radius_size)
	cv2.circle (img, center, radius + 2, (255,255,255), thickness, lineType)
	cv2.circle (img, center, radius, (0,0,0), thickness, lineType)

class point_pair_struct:
	def __init__(self, point1 = (-1,-1), point1_var = -1, point2 = (-1,-1), point2_var = -1, x_dist = 0, y_dist = 0 ):
		self.point1 = point1
		self.point1_var = point1_var
		self.point2 = point2
		self.point2_var = point2_var
		self.x_dist = x_dist
		self.y_dist = y_dist

class sac:
	def __init__(self, img1, img2):
		self.img1 = img1
		self.img2 = img2
		# size of local subimage in which dense sampling is done
		self.subimage2_size = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5 * SUBIMAGE2_SIZE_SCALE)
		# size of local subimage in which point near user selected point is searched
		self.subimage1_size = int((self.img1.shape[0] + self.img1.shape[1]) * 0.5 * SUBIMAGE1_SIZE_SCALE)
		# diameter of the meaningfull keypoint neighborhood 
		self.template_size = int((self.img2.shape[0] + self.img2.shape[1]) * 0.5  * TEMPLATE_SIZE_SCALE)

	def getPointFromRectangle(self, point, image_select):
		"""Computes point of interest in a subimage which is defined by to given points."""

		# select image on which user draw
		if image_select:
			img1 = self.img1
		else:
			img1 = self.img2

		assert 0 <= point[1] < img1.shape[0], "Point outside image"
		assert 0 <= point[0] < img1.shape[1], "Point outside image"

		# create subimage from img1 in which point is searched
		subimage1_size_half = int(self.subimage1_size/2)
		x1 = max(point[0] - subimage1_size_half, 0)
		x2 = min(point[0] + subimage1_size_half, img1.shape[1] - 1)
		y1 = max(point[1] - subimage1_size_half, 0)
		y2 = min(point[1] + subimage1_size_half, img1.shape[0] - 1)
		subimage = np.copy(img1[y1:y2, x1:x2])

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
#		rows, cols = corners.shape
#		gaus_cols = cv2.getGaussianKernel(cols, -1)
#		gaus_rows = cv2.getGaussianKernel(rows, -1)
#		gaus_matrix = gaus_rows*gaus_cols.T
#		gaus_matrix_normalized = gaus_matrix/gaus_matrix.max()
#		corners = corners * gaus_matrix_normalized
		
		# get sharpest corners
		i, j = np.where(corners == corners.max());

		# get index of corner in middle of sharpest corner array, most often there is only one entry 
		index = int(i.shape[0]/2)

		#add the start position of rectangle as offset
		return_point = (j[index] + point[0] - subimage1_size_half, i[index] + point[1] - subimage1_size_half)

		#TODO remove for build
		'''	
		corners = cv2.normalize(corners, corners, 0, 255, cv2.NORM_MINMAX)
		cv2.imshow("corners", np.uint8(corners))
		cv2.imshow("gaus", gaus_matrix_normalized)
		cv2.imshow("subimage", np.uint8(subimage_f*255))
		''' 

		var = np.var(corners)
		return return_point, var

	def getCorespondingPoint(self, point, image_select):
		"""Search for coresponding point on second image given a point in first image using template matching."""

		# select image on which user draw
		if image_select:
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
		template_size_half = int(self.template_size/2)
		x1 = max(point[0] - template_size_half, 0)
		x2 = min(point[0] + template_size_half, img1.shape[1] - 1)
		y1 = max(point[1] - template_size_half, 0)
		y2 = min(point[1] + template_size_half, img1.shape[0] - 1)
		subimage1 = np.copy(img1[y1:y2, x1:x2])

		# create subimage from img2 in which template is searched
		subimage2_size_half = int(self.subimage2_size/2)
		x1 = max(point[0] - subimage2_size_half, 0)
		x2 = min(point[0] + subimage2_size_half, img2.shape[1] - 1)
		y1 = max(point[1] - subimage2_size_half, 0)
		y2 = min(point[1] + subimage2_size_half, img2.shape[0] - 1)
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
		#TODO weight with gaussian
		minVal, maxVal, minLoc , maxLoc = cv2.minMaxLoc(template_result_1)

		var = np.var(template_result_1)
		point2 = (maxLoc[0] + template_size_half, maxLoc[1] + template_size_half)
		return_point = (x1 + int(point2[0]), y1 + int(point2[1]))

		# TODO delete for build
		'''
		myFilledCircle(subimage2_f, (int(point2[0]), int(point2[1])))
		cv2.imshow("subimage1", np.uint8(subimage1_f*255))
		cv2.imshow("subimage2", np.uint8(subimage2_f*255))
		'''

		return return_point, var

	def compute(self):
		stepsize = self.subimage1_size 
		stepsize_half = int(stepsize/2)
		x_max = min(self.img1.shape[1], self.img2.shape[1])
		y_max = min(self.img1.shape[0], self.img2.shape[0])
		# use cols and rows for fast access to neighbours
		point_pairs_cols = []
		point_pairs = []
		print stepsize
		for x in range (stepsize_half, x_max, stepsize):
			for y in range (stepsize_half, y_max, stepsize):
				point_pairs_row = []
				point1, var = self.getPointFromRectangle((x,y), True)
				point_pair = point_pair_struct(point1 = point1, point1_var = var) 
				point_pairs_row.append(copy.deepcopy(point_pair))
				point_pairs.append(copy.deepcopy(point_pair))
			point_pairs_cols.append(copy.deepcopy(point_pairs_row))

		point_pairs.sort(key=lambda x: x.point1_var, reverse=True)
		NUMBER_POINTS_IMG1 = 5000

		if NUMBER_POINTS_IMG1 < len(point_pairs):
			point_pairs = point_pairs[0:NUMBER_POINTS_IMG1-1]
	
		for point_pair in point_pairs:
			point_pair.point2, point_pair.point2_var = self.getCorespondingPoint(point_pair.point1, True)

		point_pairs.sort(key=lambda x: x.point2_var, reverse=True)
		NUMBER_POINTS_IMG2 = 2500
		if NUMBER_POINTS_IMG2 < len(point_pairs):
			point_pairs = point_pairs[0:NUMBER_POINTS_IMG2-1]
			
		count = x_dist_sum = y_dist_sum = 0
		for point_pair in point_pairs:
			count += 1
			point_pair.x_dist = (point_pair.point1[0] - point_pair.point2[0])
			point_pair.y_dist = (point_pair.point1[1] - point_pair.point2[1])
			x_dist_sum += point_pair.x_dist 
			y_dist_sum += point_pair.y_dist 
			#print (x_dir, y_dir)

		x_mean = x_dist_sum / count
		y_mean = y_dist_sum / count

		point_pairs.sort(key=lambda x: abs(x.x_dist - x_mean) +  abs(x.y_dist - y_mean))
		NUMBER_POINTS_IMG2_2 = 100
		if NUMBER_POINTS_IMG2_2 < len(point_pairs):
			point_pairs = point_pairs[0:NUMBER_POINTS_IMG2_2-1]

		for point_pair in point_pairs:
			myFilledCircle(self.img1, point_pair.point1)
			myFilledCircle(self.img2, point_pair.point2)

		cv2.namedWindow("Image 1", cv2.WINDOW_KEEPRATIO)
		cv2.namedWindow("Image 2", cv2.WINDOW_KEEPRATIO)
		cv2.imshow("Image 1", self.img1)
		cv2.imshow("Image 2", self.img2)
		cv2.resizeWindow("Image 1", 640, 1024)
		cv2.resizeWindow("Image 2", 640, 1024)

		points_img1 = []
		points_img2 = []
		for point_pair in point_pairs:
			points_img1.append(point_pair.point1)
			points_img2.append(point_pair.point2)
			
		return points_img1, points_img2
