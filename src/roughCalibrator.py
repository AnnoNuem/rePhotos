import numpy as np
import cv2

def calibrate(img1, img2, pointsImg1, pointsImg2):
	"""
	Aligns the two images with the best matching perspective transform given the two point lists.
	:param img1: Image 1
	:param img2: Image 2
	:param pointsImg1: marked points in image 1
	:param pointsImg2: coresponding points in image 2
	"""
	print pointsImg1
	print pointsImg2
	transformMatrix, status = cv2.findHomography(np.vstack(pointsImg1).astype(float), np.vstack(pointsImg2).astype(float), 0)
	#print status
	print transformMatrix
	#TODO scale of transform matrix 
	img1 = cv2.warpPerspective(img1, -0.5 * transformMatrix, (img1.shape[1], img1.shape[0]))
	img2 = cv2.warpPerspective(img2, 0.5 * transformMatrix, (img2.shape[1], img2.shape[0]))
	#cv2.imshow("bla", img1 * 0.5 + img2 * 0.5)
	cv2.imshow("bla2", img1)
	cv2.imshow("bla3", img2)
