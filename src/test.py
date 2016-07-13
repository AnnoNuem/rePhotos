import numpy as np
import cv2
import sys
import sac 

img1 = None
img2 = None
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
def on_mouse(event, x, y, flags, param):
	global drag_start, img1, img2, sac_instance, rectangle, rectangle_witdh
	if event == cv2.EVENT_LBUTTONDOWN:
		rectangle = True
		drag_start = x, y
	elif event == cv2.EVENT_MOUSEMOVE:
		if rectangle == True:
			img3 = img1.copy()
			shape = img3.shape
			width = int((shape[0] + shape[1]) * rectangle_witdh)
			cv2.rectangle(img3, drag_start, (x, y), (50,255,50), width)
			cv2.imshow("Image 1", img3)
	elif event == cv2.EVENT_LBUTTONUP and rectangle is True:
		drag_end = x,y
		rectangle = False
		# get point in rectangle and coresponding point 
		point1, point2 = 	sac_instance.getPFromRectangleACorespondingP(drag_start, drag_end)
		myFilledCircle(img1, point1)
		myFilledCircle(img2, point2)
		cv2.imshow("Image 1", img1)
		cv2.imshow("Image 2", img2)

def test():
	global img1, img2, sac_instance
	""" test method for semiautomatic point corespondence """
	if len(sys.argv) < 3:
		print ("Usage: getPoints <image_name1> <image_name2>")
		exit()

	img1, img2  = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])

	if img1 is None:
		print ("Image 1 not readable or not found")
		exit()
	if img2 is None:
		print ("Image 2 not readable or not found")
		exit()

	#create sac instance with two images	
	sac_instance = sac.sac(img1, img2)	

	cv2.namedWindow("Image 1", cv2.WINDOW_KEEPRATIO)
	cv2.setMouseCallback("Image 1", on_mouse)
	cv2.namedWindow("Image 2", cv2.WINDOW_KEEPRATIO)
	cv2.imshow("Image 1", img1)
	cv2.imshow("Image 2", img2)
	cv2.resizeWindow("Image 1", 640, 1024)
	cv2.resizeWindow("Image 2", 640, 1024)
	
	while cv2.waitKey(0) != 27:
		pass
	
	cv2.destroyAllWindows() 

if __name__ == "__main__":
    sys.exit(test())
