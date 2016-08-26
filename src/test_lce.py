import numpy as np
import cv2
from image_helpers import lce

img = cv2.imread("test_lce.jpg")

img_lce = lce(img, 101, 0.5)

cv2.imshow("img", img)
cv2.imshow("lce_result", np.uint8(img_lce * 255))

while cv2.waitKey(0) != 27:
    pass
