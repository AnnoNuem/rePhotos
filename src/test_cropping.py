import cv2
import numpy as np
from image_helpers import get_crop_indices
img = cv2.imread("test_crop.jpg")
x_min, x_max, y_min, y_max = get_crop_indices(img)
print x_min, x_max, y_min, y_max
cv2.namedWindow("img1", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("img2", cv2.WINDOW_KEEPRATIO)
cv2.imshow("img1", img[y_min: y_max, x_min: x_max,:])
cv2.imshow("img2", img)
cv2.resizeWindow("img1", 640, 480)
cv2.resizeWindow("img2", 640, 480)
while cv2.waitKey(0) != 27:
    pass
        
