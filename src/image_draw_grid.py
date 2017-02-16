import cv2
import numpy as np

def draw_grid(img, grid_points, quad_indices):
    """
    Draws line grid on given image.

    Args:
        img: Image on which to draw.
        grid_points: List of cornerpoints of the grid.
        quad_indices: List of indices of quads.
    """
    
    thickness = int((img.shape[0] + img.shape[1]) / 2500  ) + 1
    color = (255,255,255)
    x_max = img.shape[1]
    y_max = img.shape[0]

    for quad in quad_indices:
        c_p = lambda point: (min(max(int(point[0]), 0,), x_max), \
                             min(max(int(point[1]), 0), y_max))

        quad= [c_p(grid_points[quad[0]]), 
               c_p(grid_points[quad[1]]), 
               c_p(grid_points[quad[2]]), 
               c_p(grid_points[quad[3]])]

        # Only one horizontal and vertical line is drawn to reduce time. 
        cv2.line(img, quad[0], quad[1], color, thickness, 16)
        cv2.line(img, quad[0], quad[3], color, thickness, 16)
    

