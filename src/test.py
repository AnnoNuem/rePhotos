import numpy as np
import cv2
import sys
import image_sac 
from image_1_stage import compute_1_stage
from image_2_stage import compute_2_stage
from image_3_stage import compute_3_stage

img1 = None
img2 = None
img1Orig = None
img2Orig = None
img1Rough_morphed = None
img2Rough_morphed = None
stage = 1 
number_of_pointpairs_1_stage = 2
number_of_pointpairs_2_stage = 4
# points turtle tower
#points_img2 = [(268,255),(345,225),(350,291),(268,293)]
#points_img1 = [(2241,1503),(2754,1475),(2752, 1952),(2182,1973)]
# points marktplatz osna
#points_img1 = [(159, 552), (611, 845), (493, 716)]
#points_img2 = [(1050, 838), (2061, 1216), (1851, 998)]
# points backsteinneubau osnabrueck
#points_img1 = [(571, 177), (694, 182), (1471, 606), (1323, 229)]
#points_img2 = [(580, 189), (729, 201), (1582, 686), (1431, 282)]
points_img1 = []
points_img2 = []
radius_size = 0.003
rectangle_witdh = 0.0008


def my_filled_circle(img, center):
    global radius_size
    thickness = -1
    line_type = 8
    shape = img.shape
    radius = int((shape[0] + shape[1]) * radius_size)
    cv2.circle (img, center, radius, (0,0,255), thickness, line_type)


def switch_to_2_stage():
    global img1, img2, img1Orig, img2Orig, points_img1, points_img2, img1Rough_morphed, img2Rough_morphed, stage
    stage = 2
    img1 = np.copy(img1Orig)
    img2 = np.copy(img2Orig)
    print ("Roughly calibrating images...")
    img1, img2, points_img1, points_img2 = compute_2_stage(img1, img2, points_img1, points_img2)
    print ("Rough calibration done.")
    img1Rough_morphed = np.copy(img1)
    img2Rough_morphed = np.copy(img2)
    for point in points_img1:
        my_filled_circle(img1, point)
    for point in points_img2:
        my_filled_circle(img2, point) 
    cv2.namedWindow("Image 1", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Image 2", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.resizeWindow("Image 1", 640, 1024)
    cv2.resizeWindow("Image 2", 640, 1024)
    cv2.setMouseCallback("Image 1", on_mouse, True)
    cv2.setMouseCallback("Image 2", on_mouse, False)


def switch_to_3_stage():
    cv2.destroyAllWindows()
    global img1, img2, img1Orig, img2Orig, points_img1, points_img2, img1Rough_morphed, img2Rough_morphed, stage
    stage = 3
    img1 = np.copy(img1Orig)
    img2 = np.copy(img2Orig)
    print ("Roughly calibrating images...")
    img1, img2, points_img1, points_img2 = compute_2_stage(img1, img2, points_img1, points_img2)
    print ("Rough calibration done.")
    img1Rough_morphed = np.copy(img1)
    img2Rough_morphed = np.copy(img2)
    for point in points_img1:
        my_filled_circle(img1, point)
    for point in points_img2:
        my_filled_circle(img2, point) 
    cv2.namedWindow("Image 1", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Image 2", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.resizeWindow("Image 1", 640, 1024)
    cv2.resizeWindow("Image 2", 640, 1024)
    cv2.setMouseCallback("Image 1", on_mouse, True)
    cv2.setMouseCallback("Image 2", on_mouse, False)


number_of_pointpairs = 0
drag_start = None
rectangle = False
waiting_for_second_point = False
previous_point = -1
def on_mouse(event, x, y, flags, image_select):
    global drag_start, rectangle, rectangle_witdh, waiting_for_second_point, previous_point, number_of_pointpairs, stage
    if image_select:
        img1Temp = img1
        img2Temp = img2
        img1Name = "Image 1"
        img2Name = "Image 2"
    else:
        img1Temp = img2
        img2Temp = img1
        img1Name = "Image 2"
        img2Name = "Image 1"
    
    # Mouse movement, rectangle drawing
    if event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img3 = img1Temp.copy()
            shape = img3.shape
            width = int((shape[0] + shape[1]) * rectangle_witdh)
            cv2.rectangle(img3, drag_start, (x, y), (50,255,50), width)
            cv2.imshow(img1Name, img3)
    # Right mouse button
    elif event == cv2.EVENT_RBUTTONDOWN and rectangle is False:
        if (waiting_for_second_point and (image_select + previous_point) == 1):
            rectangle = True
            drag_start = x, y
            waiting_for_second_point = False
            previous_point =  -1
            number_of_pointpairs += 1
        elif not waiting_for_second_point:
            rectangle = True
            drag_start = x, y
            waiting_for_second_point = True
            previous_point = image_select
    elif event == cv2.EVENT_RBUTTONUP and rectangle is True:
        drag_end = x, y
        if image_select:
            # get point inside user drawn rectangle
            point = image_sac.get_point_from_rectangle(img1, drag_start, drag_end)
            points_img1.append(point)
        else:
            # get point inside user drawn rectangle
            point = image_sac.get_point_from_rectangle(img2, drag_start, drag_end)
            points_img2.append(point)
        rectangle = False
        my_filled_circle(img1Temp, point)
        cv2.imshow(img1Name, img1Temp)
        # check if stage is finished
        if waiting_for_second_point == False:
            if number_of_pointpairs == number_of_pointpairs_1_stage:
                switch_to_2_stage()
            elif number_of_pointpairs == number_of_pointpairs_2_stage:
                switch_to_3_stage() 
    # Left mouse button
    elif event == cv2.EVENT_LBUTTONDOWN and not waiting_for_second_point and rectangle is False and stage == 3:
        rectangle = True
        drag_start = x, y
    elif event == cv2.EVENT_LBUTTONUP and rectangle is True and not waiting_for_second_point and stage == 3:
        drag_end = x,y
        rectangle = False
        if image_select:
            # get point in rectangle and coresponding point 
            point1, point2 =    image_sac.get_p_from_rectangle_a_coresponding_p(img1, img2, drag_start, drag_end)
            points_img1.append(point1)
            points_img2.append(point2)
        else:
            # get point in rectangle and coresponding point 
            point1, point2 =    image_sac.get_p_from_rectangle_a_coresponding_p(img2, img1, drag_start, drag_end)
            points_img1.append(point2)
            points_img2.append(point1)
        number_of_pointpairs += 1
        my_filled_circle(img1Temp, point1)
        my_filled_circle(img2Temp, point2)
        cv2.imshow(img1Name, img1Temp)
        cv2.imshow(img2Name, img2Temp)
        # check if stage is finished
        if waiting_for_second_point == False:
            if number_of_pointpairs == number_of_pointpairs_1_stage:
                switch_to_2_stage()
            elif number_of_pointpairs == number_of_pointpairs_2_stage:
                switch_to_3_stage() 
            

def test():
    global img1, img2, img1Orig, img2Orig
    """
    Test method for semiautomatic point corespondence.
    """
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

    print ("Draw rectangles with LMB to search for corresponding point.")
    print ("Draw rectangles with RMB to only mark point.")
    print ("Click i.e. draw very tiny rectangle to mark point directly.")
    print ("Press Space to start morphing, ESC to quit")

    img1Orig= np.copy(img1)
    img2Orig= np.copy(img2)

    cv2.namedWindow("Image 1", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Image 2", cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("Image 1", on_mouse, True)
    cv2.setMouseCallback("Image 2", on_mouse, False)
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.resizeWindow("Image 1", 640, 1024)
    cv2.resizeWindow("Image 2", 640, 1024)

    key = 0
    while key != 32:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows() 
            exit()
    cv2.destroyAllWindows() 

    #morph images
    print ("Morphing...")
    alpha = 0.5
    steps = 3 
    img1 = np.copy(img1Rough_morphed)
    img2 = np.copy(img2Rough_morphed)
    images = compute_3_stage(img1, img2, points_img1, points_img2, alpha, steps)
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
