import cv2
import numpy as np

img = cv2.imread("bla.jpg")

img2a = np.float32(img)
img2 = cv2.cvtColor(img2a ,cv2.COLOR_BGR2GRAY) #img #np.float32(img)
img3 = cv2.normalize(img2, img2 , 0.0, 1.0, cv2.NORM_MINMAX)
img4 = img3 * 255
#for point in img4:
#	print point
cv2.imshow("image", np.uint8(img4))

while cv2.waitKey(0) != 27:
	pass
