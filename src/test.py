import numpy as np
import cv2
import sys
import sac 
import delaunay_morphing

img1 = None
img2 = None
points_img1 = []
points_img2 = []
sac_instance = None
radius_size = 0.003
rectangle_witdh = 0.0008

def myFilledCircle(img, center):
	global radius_size
	thickness = -1
	lineType = 8
	shape = img.shape
	radius = int((shape[0] + shape[1]) * radius_size)
	cv2.circle (img, center, radius, (0,0,255), thickness, lineType)

drag_start = None
rectangle = False
def on_mouse(event, x, y, flags, image_select):
	global drag_start, sac_instance, rectangle, rectangle_witdh
	if image_select:
		img1_temp = img1
		img2_temp = img2
		img1_name = "Image 1"
		img2_name = "Image 2"
	else:
		img1_temp = img2
		img2_temp = img1
		img1_name = "Image 2"
		img2_name = "Image 1"

	if event == cv2.EVENT_LBUTTONDOWN:
		rectangle = True
		drag_start = x, y
	elif event == cv2.EVENT_MOUSEMOVE:
		if rectangle == True:
			img3 = img1_temp.copy()
			shape = img3.shape
			width = int((shape[0] + shape[1]) * rectangle_witdh)
			cv2.rectangle(img3, drag_start, (x, y), (50,255,50), width)
			cv2.imshow(img1_name, img3)
	elif event == cv2.EVENT_LBUTTONUP and rectangle is True:
		drag_end = x,y
		rectangle = False
		# get point in rectangle and coresponding point 
		point1, point2 = 	sac_instance.getPFromRectangleACorespondingP(drag_start, drag_end, image_select)
		if image_select:
			points_img1.append(point1)
			points_img2.append(point2)
		else:
			points_img1.append(point2)
			points_img2.append(point1)
		myFilledCircle(img1_temp, point1)
		myFilledCircle(img2_temp, point2)
		cv2.imshow(img1_name, img1_temp)
		cv2.imshow(img2_name, img2_temp)

def test():
	global img1, img2, sac_instance
	""" test method for semiautomatic point corespondence """
	if len(sys.argv) < 3:
		print ("Usage: test <image_name1> <image_name2>")
		exit()

	img1, img2  = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])

	if img1 is None:
		print ("Image 1 not readable or not found")
		exit()
	if img2 is None:
		print ("Image 2 not readable or not found")
		exit()

	img1_copy = np.copy(img1)
	img2_copy = np.copy(img2)
	
	stepsize = sac_instance.subimage1_size
	#create sac instance with two images	

	sac_instance = sac.sac(img1, img2)	
	stepsize_half = int(stepsize/2)
	x_max = min(sac_instance.img1.shape[0], sac.img2.shape[0])
	y_max = min(sac_instance.img1.shape[1], sac.img2.shape[1])
	for x in range (stepsize_half, x_max, stepsize):
		for y in range (stepsize_half, y_max, stepsize):
			point1, point2 = sac_instance.getPointPFromP((x,y), image_select)

	#morph images
	alpha = 0.5
	steps = 3 
	images = delaunay_morphing.delaunayMorphing(img1_copy, img2_copy, points_img1, points_img2, alpha, steps)
	cv2.namedWindow("Image 1 morphed", cv2.WINDOW_KEEPRATIO)
	cv2.namedWindow("Image 2 morphed", cv2.WINDOW_KEEPRATIO)
	cv2.namedWindow("Images blended", cv2.WINDOW_KEEPRATIO)
	cv2.imshow("Image 1 morphed", images[0])
	cv2.imshow("Image 2 morphed", images[2])
	cv2.imshow("Images blended", images[1])
	cv2.resizeWindow("Image 1 morphed", 640, 1024)
	cv2.resizeWindow("Image 2 morphed", 640, 1024)
	cv2.resizeWindow("Images blended", 640, 1024)
	while cv2.waitKey(0) != 27:
		pass
	cv2.destroyAllWindows() 
	

if __name__ == "__main__":
    sys.exit(test())
