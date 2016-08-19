import numpy as np


def crop(img):
    pass

def weighted_average_point(point1, point2, alpha):
    """
    Return the average point between two points weighted by alpha.
    """
    x = int((1 - alpha) * point1[0] + alpha * point2[0])
    y = int((1 - alpha) * point1[1] + alpha * point2[1])
    return (x,y)


def get_corners(img, img2, points_img1, points_img2, x_crop, y_crop, alpha):
    """Adds the corners and middle point of edges to pointlists.
    Finds the user selectet points which are nearest to the four corners and the
    four middle points of the edges of the image. Computes the delta between
    them and their coresponding points. Adds corner points and middle points of
    edges to the point lists and offsets them using the computet delta values.
    Returns the global max and minima used for cropping
    """

    x_max = min(img.shape[1], img2.shape[1]) - 1
    y_max = min(img.shape[0], img2.shape[0]) - 1
    x_mean = int(x_max / 2)
    y_mean = int(y_max / 2)
    corners_img1 = []
    corners_img2 = []

    # left middle
    p_min_mean, i_min_mean = min(((val, idx) for (idx, val) in enumerate(points_img1)),
                                key=lambda p: (p[0])[0] + abs(y_mean - (p[0])[1]))
    delta_y_half = int((p_min_mean[1] - (points_img2[i_min_mean])[1]) / 2)
    delta_x_half = int((p_min_mean[0] - (points_img2[i_min_mean])[0]) / 2)
    corners_img1.append((0 + abs(delta_x_half) + delta_x_half, p_min_mean[1] + delta_y_half))
    corners_img2.append((0 + abs(delta_x_half) - delta_x_half, p_min_mean[1] - delta_y_half))
    average_point =  weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_x_min =  average_point[0]

    # right middle
    p_max_mean, i_max_mean = min(((val, idx) for (idx, val) in enumerate(points_img1)),
                                key=lambda p: (x_max - (p[0])[0]) + abs(y_mean - (p[0])[1]))
    delta_y_half = int((p_max_mean[1] - (points_img2[i_max_mean])[1]) / 2)
    delta_x_half = int((p_max_mean[0] - (points_img2[i_max_mean])[0]) / 2)
    corners_img1.append((x_max - abs(delta_x_half) + delta_x_half, p_max_mean[1] + delta_y_half))
    corners_img2.append((x_max - abs(delta_x_half) - delta_x_half, p_max_mean[1] - delta_y_half))
    average_point = weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_x_max = average_point[0] 

    # top middle
    p_mean_min, i_mean_min = min(((val, idx) for (idx, val) in enumerate(points_img1)),
                                key=lambda p: abs(x_mean - (p[0])[0]) + (p[0])[1])
    delta_x_half = int((p_mean_min[0] - (points_img2[i_mean_min])[0]) / 2)
    delta_y_half = int((p_mean_min[1] - (points_img2[i_mean_min])[1]) / 2)
    corners_img1.append((p_mean_min[0] + delta_x_half, 0 + abs(delta_y_half) + delta_y_half))
    corners_img2.append((p_mean_min[0] - delta_x_half, 0 + abs(delta_y_half) - delta_y_half))
    average_point = weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_y_min = average_point[1] 

    # bottom middle
    p_mean_max, i_mean_max = min(((val, idx) for (idx, val) in enumerate(points_img1)),
                                key=lambda p: abs(x_mean - (p[0])[0]) + (y_max - (p[0])[1]))
    delta_x_half = int((p_mean_max[0] - (points_img2[i_mean_max])[0]) / 2)
    delta_y_half = int((p_mean_max[1] - (points_img2[i_mean_max])[1]) / 2)
    corners_img1.append((p_mean_max[0] + delta_x_half, y_max - abs(delta_y_half) + delta_y_half))
    corners_img2.append((p_mean_max[0] - delta_x_half, y_max - abs(delta_y_half) - delta_y_half))
    average_point = weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_y_max = average_point[1]

    # bottom left
    p_min_max, i_min_max = max(((val, idx) for (idx, val) in enumerate(points_img1)), 
                                key=lambda p: (x_max - (p[0])[0]) + (p[0])[1])
    delta_y_half = int((p_min_max[1] - (points_img2[i_min_max])[1]) / 2)
    delta_x_half = int((p_min_max[0] - (points_img2[i_min_max])[0]) / 2)
    corners_img1.append((0 + abs(delta_x_half) + delta_x_half, y_max - abs(delta_y_half) + delta_y_half))
    corners_img2.append((0 + abs(delta_x_half) - delta_x_half, y_max - abs(delta_y_half) - delta_y_half))
    average_point = weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_x_min = average_point[0] if average_point[0] > global_x_min else global_x_min
    global_y_max = average_point[1] if average_point[1] < global_y_max else global_y_max

    # bottom right
    p_max_max, i_max_max = max(((val, idx) for (idx, val) in enumerate(points_img1)), 
                                key=lambda p: (p[0])[0] + (p[0])[1])
    delta_y_half = int((p_max_max[1] - (points_img2[i_max_max])[1]) / 2)
    delta_x_half = int((p_max_max[0] - (points_img2[i_max_max])[0]) / 2)
    corners_img1.append((x_max - abs(delta_x_half) + delta_x_half, y_max - abs(delta_y_half) + delta_y_half))
    corners_img2.append((x_max - abs(delta_x_half) - delta_x_half, y_max - abs(delta_y_half) - delta_y_half))
    average_point = weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_x_max = average_point[0] if average_point[0] < global_x_max else global_x_max
    global_y_max = average_point[1] if average_point[1] < global_y_max else global_y_max

    # top right
    p_max_min, i_max_min = max(((val, idx) for (idx, val) in enumerate(points_img1)), 
                                key=lambda p: (p[0])[0] + (y_max - (p[0])[1]))
    delta_y_half = int((p_max_min[1] - (points_img2[i_max_min])[1]) / 2)
    delta_x_half = int((p_max_min[0] - (points_img2[i_max_min])[0]) / 2)
    corners_img1.append((x_max - abs(delta_x_half) + delta_x_half, 0 + abs(delta_y_half) + delta_y_half))
    corners_img2.append((x_max - abs(delta_x_half) - delta_x_half, 0 + abs(delta_y_half) - delta_y_half))
    average_point = weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_x_max = average_point[0] if average_point[0] < global_x_max else global_x_max
    global_y_min = average_point[1] if average_point[1] > global_y_min else global_y_min

    # top left
    p_min_min, i_min_min = min(((val, idx) for (idx, val) in enumerate(points_img1)), 
                                key=lambda p: (p[0])[0] + (p[0])[1])
    delta_y_half = int((p_min_min[1] - (points_img2[i_min_min])[1]) / 2)
    delta_x_half = int((p_min_min[0] - (points_img2[i_min_min])[0]) / 2)
    corners_img1.append((0 + abs(delta_x_half) + delta_x_half, 0 + abs(delta_y_half) + delta_y_half))
    corners_img2.append((0 + abs(delta_x_half) - delta_x_half, 0 + abs(delta_y_half) - delta_y_half))
    average_point = weighted_average_point(corners_img1[-1], corners_img2[-1], alpha)
    global_x_min = average_point[0] if average_point[0] > global_x_min else global_x_min
    global_y_min = average_point[1] if average_point[1] > global_y_min else global_y_min

    points_img1 += corners_img1
    points_img2 += corners_img2
    return global_x_min, global_x_max, global_y_min, global_y_max


def scale(img1, img2, points_img1, points_img2):
    """
    Prescales images and points to allow delaunay morphing of images of different sizes.
    Upsacles the smaller image
    """
    y_size_img1, x_size_img1, z_size_img1 = img1.shape
    y_size_img2, x_size_img2, z_size_img2 = img2.shape
    x_scale_factor = float(x_size_img1)/ float(x_size_img2)
    y_scale_factor = float(y_size_img1)/ float(y_size_img2)

    # images are of same size
    if x_size_img1 == x_size_img2 and y_size_img1 == y_size_img2:
        return img1, img2, points_img1, points_img2, x_size_img1, y_size_img1

    # image 1 is bigger
    elif x_size_img1 >= x_size_img2 and y_size_img1 >= y_size_img2:
        temp_img = np.zeros(img1.shape, dtype=img1.dtype)
        temp_points = []
        # x scale is smaller
        if x_scale_factor < y_scale_factor:
            img2 = cv2.resize(img2,  (0,0), fx=x_scale_factor, fy=x_scale_factor, interpolation=cv2.INTER_CUBIC)
            # save how big img2 really is for cropping. Later it will have size of img1 with zeros filling remaining space
            y_crop = img2.shape[0]
            x_crop = img2.shape[1]
            for point in points_img2:
                x = point[0] * x_scale_factor
                y = point[1] * x_scale_factor
                temp_points.append((x,y))
        # y scale is smaller
        else:
            img2 = cv2.resize(img2, (0,0), fx=y_scale_factor, fy=y_scale_factor, interpolation=cv2.INTER_CUBIC)
            x_crop = img2.shape[1]
            y_crop = img2.shape[0]
            for point in points_img2:
                x = point[0] * y_scale_factor
                y = point[1] * y_scale_factor
                temp_points.append((x,y))
        temp_img[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]] = img2
        points_img2 = temp_points
        img2 = temp_img

    #image 1 is smaller
    elif x_size_img1 <= x_size_img2 and y_size_img1 <= y_size_img2:
        temp_img = np.zeros(img2.shape, dtype=img2.dtype)
        temp_points = []
        # x scale is smaller. we need the inverse
        if x_scale_factor > y_scale_factor:
            img1 = cv2.resize(img1, (0,0), fx=(1/x_scale_factor), fy=(1/x_scale_factor), interpolation=cv2.INTER_CUBIC)
            # use *-1 to indicate that image 1 is smaller
            y_crop = -img1.shape[0]
            x_crop = -img1.shape[1]
            for point in points_img1:
                x = point[0] * 1/x_scale_factor
                y = point[1] * 1/x_scale_factor
                temp_points.append((x,y))
        # y scale is smaller
        else:
            img1 = cv2.resize(img1, (0,0), fx=1/y_scale_factor, fy=1/y_scale_factor, interpolation=cv2.INTER_CUBIC)
            x_crop = -img1.shape[1]
            y_crop = -img1.shape[0]
            for point in points_img1:
                x = point[0] * 1/y_scale_factor
                y = point[1] * 1/y_scale_factor
                temp_points.append((x,y))
        temp_img[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]] = img1
        points_img1 = temp_points
        img1 = temp_img

    # images size relations are not the same i.e. x_scale < 0 and y_scale > 0 or vice versa
    else:
        temp_img = np.zeros((max(y_size_img1, y_size_img2), max(x_size_img1, x_size_img2), max(z_size_img1, z_size_img2)), dtype=img1.dtype)
        temp_img2 = np.copy(temp_img)
        x_crop = -x_size_img1 if x_size_img1 < x_size_img2 else x_size_img2
        y_crop = -y_size_img1 if y_size_img1 < y_size_img2 else y_size_img2
        temp_img[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]] = img1
        img1 = temp_img
        temp_img2[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]] = img2
        img2 = temp_img2

    return img1, img2, points_img1, points_img2, x_crop, y_crop

