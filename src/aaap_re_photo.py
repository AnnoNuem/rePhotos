#!/usr/bin/env python
"""Mainscript for image alignment in the re.photos project.
Two images which names are given as parameters at program start are loaded and
scaled to the same size. In a two stage approach the user aligns the images. In
the first stage the user draws rectangles on either of the two images. A 
probable corner point inside the rectangle meant by the user is automatically 
selected as well as a corresponding point in the second image. Alternatively the
user can draw points directly. With four pointpairs a perspective transform is 
performed aligning the images roughly. In the second stage the user draws lines 
instead of point and as-affine-as-possible warping is used to tune the image 
alignment. After that the images are cropped to the maximal possible size.
"""
from __future__ import print_function
import argparse
import os
import math
import cv2 
import numpy as np
import sys
import image_lines as i_l
import image_io as i_io
import image_helpers as i_h
from numpy.linalg import inv
from image_aaap_main import aaap_morph
from image_sac import getPointFromPoint
from image_sac import getPointFromRectangle
from image_sac import getCorespondingPoint
from image_perspective_alignment import perspective_align
from image_perspective_alignment import transform_lines

line_file_name_end = "line_file.json"
point_file_name_end = "point_file.json"
    
ps = lambda p, sf: (p[0]*sf, p[1]*sf)
ls = lambda l, sf: [v * sf for v in l]

drag_start = (0,0)
number_of_points = 0
def onMouse_stage_one(event, x, y, flags,
    (img_d, img_d_clean, img_orig, scale, points, win_name, color,
     img2_d, img2_d_clean, img2_orig, scale2, points2, win_name2, color2)):
    """Mousecallback function for first stage user input.
    User input is drawn on resized images. Search for point and corresponding
    point is done on original sized images. 
    Custom parameters are given as one tuple.

    Parameters
    ----------
    event : int
        Mouse event. 
    x : int
        X-coordinate of mouse pointer.
    y : int
        Y-coordinate of mouse pointer.
    flags : int
        Type of mouse event.
    img_d : ndarray
        Resized image on which points are drawn.
    img_d_clean : ndarray
        Resized image to reset img_d if points are removed.
    img_orig : ndarray
        Unscaled image in which point is searched.
    scale : float
        Scale of img_d with respect to imig_orig.
    points : list
        List of points.
    win_name : string
        Name of window in which image is shown.
    color : tuple
        Color of drawn points.
    img2_d : ndarray
        Resized image for corresponding point on which points are drawn.
    img2_d_clean : ndarray
        Resized image for corresponding point to reset img_d if points 
        are removed.
    img2_orig : ndarray
        Unscaled image for corresponding point in which corresponding
        point is searched.
    scale2 : float
        Scale of img2_d with respect to imig2_orig.
    points2 : list
        List of corresponding points.
    win_name2 : string
        Name of window in which image2 is shown.
    color2 : tuple
        Color of corresponding points.

    Returns
    -------

    """
    global drag_start, number_of_points

    img_tmp = np.copy(img_d)
    if event == cv2.EVENT_LBUTTONUP and len(points) < 4:
        points.append((x/scale,y/scale))
        number_of_points = number_of_points + 1
        i_h.draw_circle(img_d, (x,y), color)
        cv2.imshow(win_name, img_d)
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) < 4:
        drag_start = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON\
         and len(points) < 4:
        i_h.draw_rectangle(img_tmp, drag_start, (x,y), color)
        cv2.imshow(win_name, img_tmp)
    elif event == cv2.EVENT_RBUTTONUP and len(points) < 4:
        point = getPointFromRectangle(img_orig[:,:,0:3], (drag_start[0]/scale, 
                        drag_start[1]/scale), (x/scale,y/scale))
        i_h.draw_circle(img_d, (point[0]*scale, point[1]*scale), color)
        cv2.imshow(win_name, img_d)
        points.append(point)
        number_of_points = number_of_points + 1

        point2 = getCorespondingPoint(img_orig[:,:,:], img2_orig[:,:,:], point)
        if point2 is not None:
            i_h.draw_circle(img2_d, (point2[0]*scale2, point2[1]*scale2), color2)
            cv2.imshow(win_name2, img2_d)
            points2.append(point2)
            number_of_points = number_of_points + 1
    elif event == cv2.EVENT_MBUTTONUP and len(points) > 0:
        del points[-1]
        img_d[:] = img_d_clean[:]
        number_of_points -= 1
        for point in points:
            i_h.draw_circle(img_d, (point[0]*scale, point[1]*scale), color)
        cv2.imshow(win_name, img_d)


def onMouse_stage_two(event, x, y, flags, 
        (img_d, img_d_clean, img_orig, scale, lines, win_name, color,
         img2_d, img2_d_clean, img2_orig, scale2, lines2, win_name2, color2)):
    """Mousecallback function for second stage user input.
    User input is drawn on resized images. Search for line and corresponding
    line is done on original sized images. 
    Custom parameters are given as one tuple.
    
    Parameters
    ----------
    event : int
        Mouse event. 
    x : int
        X-coordinate of mouse pointer.
    y : int
        Y-coordinate of mouse pointer.
    flags : int
        Type of mouse event.
    img_d : ndarray
        Resized image on which lines are drawn.
    img_d_clean : ndarray
        Resized image to reset img_d if lines are removed.
    img_orig : ndarray
        Unscaled image in which lines is searched.
    scale : float
        Scale of img_d with respect to imig_orig.
    lines : list
        List of lines.
    win_name : string
        Name of window in which image is shown.
    color : tuple
        Color of drawn lines.
    img2_d : ndarray
        Resized image for corresponding lines on which lines are drawn.
    img2_d_clean : ndarray
        Resized image for corresponding lines to reset img_d if lines 
        are removed.
    img2_orig : ndarray
        Unscaled image for corresponding line in which corresponding
        line is searched.
    scale2 : float
        Scale of img2_d with respect to imig2_orig.
    points2 : list
        List of corresponding lines.
    win_name2 : string
        Name of window in which image2 is shown.
    color2 : tuple
        Color of corresponding lines.

    Returns
    -------

    """
    global drag_start, point_stage, number_of_points

    img_d_tmp = np.copy(img_d)
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = ps((x, y), 1/scale)

    elif event == cv2.EVENT_RBUTTONDOWN:
        drag_start = getPointFromPoint(img_orig[:,:,0:3], ps((x,y), 1/scale))

    elif event == cv2.EVENT_MOUSEMOVE and (flags==cv2.EVENT_FLAG_LBUTTON\
         or flags==cv2.EVENT_FLAG_RBUTTON):
        i_h.draw_line(img_d_tmp, ps(drag_start, scale), (x,y), color, -1)
        cv2.imshow(win_name, img_d_tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        distance = math.sqrt((x/scale-drag_start[0])**2 +
                             (y/scale-drag_start[1])**2)
        # if short line, assume user wanted to add point. Add two short lines
        if distance < 4:
            lines.append([drag_start[0], drag_start[1]-1, drag_start[0], 
                         drag_start[1]+1])
            line_s = ls(lines[-1], scale)
            i_h.draw_line(img_d, (line_s[0], line_s[1]), (line_s[2], line_s[3]), 
                          color, len(lines))
            lines.append([drag_start[0]-1, drag_start[1], drag_start[0]+1, 
                         drag_start[1]])
            line_s = ls(lines[-1], scale)
            i_h.draw_line(img_d, (line_s[0], line_s[1]), (line_s[2], line_s[3]), 
                          color, len(lines))
        else:
            lines.append([drag_start[0], drag_start[1], x/scale, y/scale])
            i_h.draw_line(img_d, ls(drag_start, scale),(x,y), color, len(lines))
        cv2.imshow(win_name, img_d)

    elif event == cv2.EVENT_RBUTTONUP:
        drag_end = getPointFromPoint(img_orig[:,:,0:3], (x/scale, y/scale))
        line = i_l.get_line(drag_start, drag_end, img_orig)
        lines.append(line)
        line_s = ls(line, scale)
        i_h.draw_line(img_d,(line_s[0], line_s[1]), (line_s[2], line_s[3]), 
                      color, len(lines))
        cv2.imshow(win_name, img_d)

        # get coresponding line
        line2 = i_l.get_corresponding_line(img_orig, img2_orig, line)
        if line2 is not None:
            lines2.append(line2)
            line2_s = ls(line2, scale2)
            i_h.draw_line(img2_d, (line2_s[0], line2_s[1]), (line2_s[2], line2_s[3]),
                          color2, len(lines2))
            cv2.imshow(win_name2, img2_d)

    elif event == cv2.EVENT_MBUTTONUP and len(lines) > 0:
        del lines[-1]
        # np.copy creates new array, local img points to new array, 
        # main img still points to old img with lines
        img_d[:] = img_d_clean
        i = 1
        for line in lines:
            line_s = ls(line, scale)
            i_h.draw_line(img_d, (line_s[0], line_s[1]), (line_s[2], line_s[3]), 
                          color, i)
            i+=1
        cv2.imshow(win_name, img_d)


def init():
    """ Processes command line parameters, reads images, lines and points.
    
    Parameters
    ----------
    
    Returns
    -------
    src_img : ndarray
        Image which is warped.
    dst_img : ndarray
        Image to which the src_img is warped.
    src_lines : list
        List of lines in src_img read from line file. If line file not found or
        command line parameter *--line_file* not given an empty list is returned.
    dst_lines : list
        List of lines in dst_img read from line file. If line file not found or
        command line parameter *--line_file* not given an empty list is returned.
    src_points : list
        List of points in src_img read from point file. If point file not found or
        command line parameter *--point_file* not given an empty list is returned.
    dst_points : list
        List of points in dst_img read from point file. If point file not found or
        command line parameter *-point_file* not given an empty list is returned.
    args : Namespace
        Command line arguments.

    """
    # Parse Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("src_name", help="source image")
    parser.add_argument("dst_name", help="destination image")
    parser.add_argument("-l", "--line_file", help="read lines from and save lines\
                        to line file", action="store_true")
    parser.add_argument("-p", "--point_file", help="read points from and save points\
                        to point file", action="store_true")
    parser.add_argument("-sf", "--show_frame", help="do not crop the resulting\
                        images but show a frame where images would be cropped",
                        action="store_true")
    parser.add_argument("--scale_factor", help="Images are scaled with this\
                         factor for better quality or faster computation", 
                         default=1, type=int)
    parser.add_argument("-sg", "--show_grid", help="draw aaap warp grid on\
                        result images", action="store_true")
    #TODO make float scale avaiable 
    parser.add_argument("-sui", "--show_user_input", help="Show user drawn points\
                        and lines in result images", action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-w", "--write", help="write result images to files in\
                        resultsfolder", action="store_true")
    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        i_h.set_verbose(True)

    i_h.vprint("Starting...")

    # Read images
    src_img = cv2.imread(args.src_name)
    dst_img = cv2.imread(args.dst_name)
    if src_img is None:
        print("Image 1 not readable or not found")
        exit()
    if dst_img is None:
        print("Image 2 not readable or not found")
        exit()
    
    if args.line_file:
        line_file_name = args.src_name.rsplit('.', 1)[0] + "_" + line_file_name_end
        src_lines, dst_lines = i_io.read_lines(line_file_name)
    else:
        src_lines = []
        dst_lines = []

    if args.point_file:
        point_file_name = args.src_name.rsplit('.', 1)[0] + "_" + point_file_name_end
        src_points, dst_points = i_io.read_points(point_file_name)
    else:
        src_points = []
        dst_points = []

    if args.write:
        if not os.path.exists('results'):
            os.makedirs('results')
        args.filename_prefix = 'results/' +\
            (args.src_name.rsplit('/',1)[-1]).rsplit('.',1)[0] +\
            '_' + (args.dst_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + '_'

    return src_img, dst_img, src_lines, dst_lines, src_points, dst_points, args


def stage_one(src_img, dst_img, src_points, dst_points, src_lines, dst_lines, args):
    """First stage. User drawn points are used for perspective alignment.

    Parameters
    ----------
    src_img : ndarray
        Image on which aaap-warping will be later performed.
    dst_img : ndarray
        Image which is only perspective transformed.
    src_points : list
        Points in src_img.
    dst_points : list
        Points in dst_img. 
    src_lines : list
        Lines in src_img.
    dst_lines : list
        Lines in dst_img
    args : Namespace
        Parameters.

    Returns
    -------
    src_img : ndarray
        Perspective transformed src_img.
    dst_img : ndarray
        Perspective transformed dst_img.
    src_points : list
        Perspective transformed points in src_img.
    dst_points : list
        Perspective transformed points in dst_img.
    src_lines : list
        Perspective transformed lines in src_img.
    dst_lines : list
        Perspective transformed lines in dst_img.
    src_transform_matrix : ndarray
        Perspective transform matrix of src image, lines and points.
    dst_transform_matrix : ndarray
        Perspective transform matrix of dst image, lines and points.
    stage_one_success : bool
        True if four point pairs where avaiable to perform perspective
        transform of src and dst images, lines and points, else False.

    """
    print("First Stage: Draw four points for initial perspective transform.\
           \nDrawing less than four points omits the first stage.\
           \nLMB: Draw point.\
           \nRMB: Draw rectangle. Best point in rectangle is computed.\
           \nMMB: Delete last point in active window.\
           \nSpace: Go to second stage.\
           \nESC: Quit program.\n")

    src_img_d, src_scale_d = i_h.show_image(src_img, name='src_img')
    dst_img_d, dst_scale_d = i_h.show_image(dst_img, name='dst_img')
    src_img_d_clean = np.copy(src_img_d)
    dst_img_d_clean = np.copy(dst_img_d)
    
    for point in src_points:
        i_h.draw_circle(src_img_d, ps(point ,src_scale_d), (255,255,0))
    for point in dst_points:
        i_h.draw_circle(dst_img_d, ps(point, dst_scale_d), (0,0 ,255))
    cv2.imshow('src_img', src_img_d)
    cv2.imshow('dst_img', dst_img_d)

    cv2.setMouseCallback("src_img", onMouse_stage_one, 
                         (src_img_d, src_img_d_clean, src_img, src_scale_d, 
                          src_points, 'src_img', (255,255,0),
                          dst_img_d, dst_img_d_clean, dst_img, dst_scale_d, 
                          dst_points, 'dst_img', (0,0,255)))
    cv2.setMouseCallback("dst_img", onMouse_stage_one, 
                         (dst_img_d, dst_img_d_clean, dst_img, dst_scale_d, 
                          dst_points, 'dst_img', (0,0,255),
                          src_img_d, src_img_d_clean, src_img, src_scale_d, 
                          src_points, 'src_img', (255,255,0)))

    # wait till user has drawn all points 
    key = 0
    while key != 32: 
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            exit()
        
    cv2.destroyAllWindows()

    # write points to file 
    if args.point_file:
        point_file_name = args.src_name.rsplit('.', 1)[0] + "_" + point_file_name_end
        # scale points to original file size and save
        src_points_orig = [ps(point, 1/float(args.src_scale)) for point in src_points]
        dst_points_orig = [ps(point, 1/float(args.dst_scale)) for point in dst_points]
        i_io.write_points(src_points_orig, dst_points_orig, point_file_name)

    # perspective alignment
    if len(src_points) + len(dst_points) == 8:
        i_h.vprint("Perspective transform...")
    
        if args.show_user_input:
            # workarround with copy since opencv python wrapper throws error on
            # cv2.line(img[:,:,0:3]....
            src_tmp = src_img[:,:,0:3].copy()
            for point in src_points:
                i_h.draw_circle(src_tmp, point, (255,255,0))
            src_img[:,:,0:3] = src_tmp
            dst_tmp = dst_img[:,:,0:3].copy()
            for point in dst_points:
                i_h.draw_circle(dst_tmp, point, (0,0,255))
            dst_img[:,:,0:3] = dst_tmp
        src_img, dst_img, src_points, dst_points, src_lines, dst_lines, \
            src_transform_matrix, dst_transform_matrix = perspective_align(
                src_img, dst_img, src_points, dst_points, src_lines, dst_lines, 
                alpha=0.5)

        transform_overlay = np.uint8(cv2.addWeighted(src_img, 0.5, dst_img, 0.5, 0))
        i_h.show_image(src_img, name='src_transformed')
        i_h.show_image(dst_img, name='dst_transformed')
        i_h.show_image(transform_overlay, name='transform_overlay')

        # write2disk
        if args.write:
            if args.scale_factor == 1:
                cv2.imwrite(args.filename_prefix + 'src_perspective_transform.jpg', 
                            src_img[:,:,0:3])
                cv2.imwrite(args.filename_prefix + 'dst_perspective_transform.jpg', 
                            dst_img[:,:,0:3])
                cv2.imwrite(args.filename_prefix + 'overlay_perspective_transform.jpg', 
                            transform_overlay[:,:,0:3])
            else:
                if 1/args.src_scale > 1:
                    method = cv2.INTER_LINEAR
                else:
                    method = cv2.INTER_AREA
                cv2.imwrite(args.filename_prefix + 'src_perspective_transform.jpg', 
                            cv2.resize(src_img[:,:,0:3], (0,0), fx=1/float(args.src_scale), 
                            fy= 1/float(args.src_scale), interpolation=method))
                cv2.imwrite(args.filename_prefix + 'dst_perspective_transform.jpg', 
                            cv2.resize(dst_img[:,:,0:3], (0,0), fx=1/float(args.dst_scale), 
                            fy= 1/float(args.dst_scale), interpolation=method))
                cv2.imwrite(args.filename_prefix + 'overlay_perspective_transform.jpg', 
                            cv2.resize(transform_overlay[:,:,0:3], (0,0), fx=1/float(args.src_scale), 
                            fy= 1/float(args.src_scale), interpolation=method))

        print("Perspective transform done.\
              \nPress SPACE to continue to aaap warping. ESC to exit.\n")

        key = 0
        while key != 32 :
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit()
        stage_one_success = True
    else:
        i_h.vprint("Not enough points for perspective transform. Skipping.\n")
        src_transform_matrix = None
        dst_transform_matrix = None 
        stage_one_success = False

    cv2.destroyAllWindows()
    return src_img, dst_img, src_points, dst_points, src_lines, dst_lines,\
           src_transform_matrix, dst_transform_matrix, stage_one_success
    

def stage_two(src_img, dst_img, src_points, dst_points, src_lines, dst_lines, 
              src_transform_matrix, dst_transform_matrix, stage_one_success, args):
    """Second stage. User drawn lines are used for aaap-warping.

    Parameters
    ----------
    src_img : ndarray
        Image on which aaap-warping will be later performed.
    dst_img : ndarray
        Image to which src_img shall be warped.
    src_points : list
        Points in src_img.
    dst_points : list
        Points in dst_img. 
    src_lines : list
        Lines in src_img.
    dst_lines : list
        Lines in dst_img
    src_transform_matrix : ndarray
        Perspective transform matrix of src image, lines and points
        from first stage.
    dst_transform_matrix : ndarray
        Perspective transform matrix of dst image, lines and points
        from first stage.
    stage_one_success : bool
        True if first stage was successful, else False.
    args : Namespace
        Parameters.

    Returns
    -------
    
    """
    print("Second Stage: Draw lines for fine grain aaap warping.\
          \nDrawing no line omits the second stage.\
          \nLMB: Draw Line\
          \nMMB: Delete last line in active Window\
          \nRMB: Drawing line with RMB finds nearest line\
          \nSpace: Start morphing\nEsc: Quit program")

    src_img_d, src_scale_d = i_h.show_image(src_img, name='src_img_transformed')
    dst_img_d, dst_scale_d = i_h.show_image(dst_img, name='dst_img_transformed')
    src_img_d_clean = np.copy(src_img_d)
    dst_img_d_clean = np.copy(dst_img_d)
    
    i = 0
    for line in src_lines:
        line_s = ls(line, src_scale_d)
        i_h.draw_line(src_img_d, (line_s[0], line_s[1]), (line_s[2], line_s[3]), 
                      (255,255,0), i)
        i += 1
    i = 0
    for line in dst_lines:
        line_s = ls(line, dst_scale_d)
        i_h.draw_line(dst_img_d, (line_s[0], line_s[1]), (line_s[2], line_s[3]), 
                      (0,0 ,255), i)
        i += 1
    i_h.show_image(src_img_d, name="src_img_transformed")
    i_h.show_image(dst_img_d, name="dst_img_transformed")

    cv2.setMouseCallback("src_img_transformed", onMouse_stage_two, 
                         (src_img_d, src_img_d_clean, src_img, src_scale_d, 
                          src_lines, 'src_img_transformed', (255,255,0),
                          dst_img_d, dst_img_d_clean, dst_img, dst_scale_d, 
                          dst_lines, 'dst_img_transformed', (0,0,255)))
    cv2.setMouseCallback("dst_img_transformed", onMouse_stage_two, 
                         (dst_img_d, dst_img_d_clean, dst_img, dst_scale_d, 
                          dst_lines, 'dst_img_transformed', (0,0,255),
                          src_img_d, src_img_d_clean, src_img, src_scale_d, 
                          src_lines, 'src_img_transformed', (255,255,0)))

    # wait till user has drawn all lines
    key = 0
    while key != 32 :
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()

    # write lines
    if args.line_file:
        line_file_name = args.src_name.rsplit('.', 1)[0] + "_" + line_file_name_end
        # scale and transform lines back to original file size
        if src_transform_matrix is not None:
            src_lines_orig = [ls(line, 1/float(args.src_scale)) for line in 
                              transform_lines(src_lines, inv(src_transform_matrix))]
        else:
            src_lines_orig = [ls(line, 1/float(args.src_scale)) for line in src_lines]
        if dst_transform_matrix is not None:
            dst_lines_orig = [ls(line, 1/float(args.dst_scale)) for line in 
                              transform_lines(dst_lines, inv(dst_transform_matrix))]
        else:
            dst_lines_orig = [ls(line, 1/float(args.dst_scale)) for line in dst_lines]
        i_io.write_lines(src_lines_orig, dst_lines_orig, line_file_name)

    # aaap warping perspective transformed images
    if len(src_lines) > 0:    
        if args.show_user_input:
            # workarround with copy since opencv python wrapper throws error on
            # cv2.line(img[:,:,0:3]....
            src_tmp = src_img[:,:,0:3].copy()
            for line in src_lines:
                i_h.draw_line(src_tmp, (line[0], line[1]), (line[2], line[3]), (255,255,0))
            src_img[:,:,0:3] = src_tmp
            dst_tmp = dst_img[:,:,0:3].copy()
            for line in dst_lines:
                i_h.draw_line(dst_img[:,:,0:3].copy(), (line[0], line[1]), (line[2], line[3]), (0,0,255))
            dst_img[:,:,0:3] = dst_tmp
        # add each point from perspective transform as two short orthagonal lines
        # to keep these points fixed during warping
        if stage_one_success:
            for point in src_points:
                src_lines.append([point[0], point[1], point[0]+1, point[1]])
                src_lines.append([point[0], point[1], point[0], point[1]+1])
            for point in dst_points:
                dst_lines.append([point[0], point[1], point[0]+1, point[1]])
                dst_lines.append([point[0], point[1], point[0], point[1]+1])

        # aaap morph
        src_img_morphed, src_img_cropped, dst_img_cropped = aaap_morph(src_img, 
            dst_img, src_lines, dst_lines, line_constraint_type=2, grid_size=20, 
            scale_factor=args.scale_factor, show_frame=args.show_frame, 
            draw_grid_f=args.show_grid)

        # compute overlay
        overlay_morphed = cv2.addWeighted(dst_img_cropped, 0.5, src_img_morphed, 0.5, 0)
        overlay_orig = cv2.addWeighted(dst_img_cropped, 0.5, src_img_cropped, 0.5, 0)
        
        # display
        i_h.show_image(overlay_morphed, 'overlay')
        i_h.show_image(overlay_orig, 'overlay_orig')
        i_h.show_image(src_img_morphed, 'src_morphed')
        
        # write2disk
        if args.write:
            if len(src_points) + len(dst_points) == 8:
                filename_prefix = args.filename_prefix + '_aaap_with_perspective__'
            else:
                filename_prefix = args.filename_prefix + '_aaap_without_perspective__'
            cv2.imwrite(filename_prefix + 'src_morphed.jpg', src_img_morphed)
            cv2.imwrite(filename_prefix + 'src_cropped.jpg', src_img_cropped)
            cv2.imwrite(filename_prefix + 'dst_cropped.jpg', dst_img_cropped)
            cv2.imwrite(filename_prefix + 'overlay_aaap.jpg', overlay_morphed)
    else:
        print("No lines for aaap warping. Skipping")

    
def main():
    """First function to be called. Initializes programm and switches stages."""
    global point_stage, number_of_points

    # Initialize
    src_img, dst_img, src_lines, dst_lines, src_points, dst_points, args = init()
    
    src_img = np.float32(src_img)
    dst_img = np.float32(dst_img)

    # Scale images
    i_h.vprint("Scaling...")
    # Add alpha channel for cropping
    src_img = np.concatenate([src_img, np.ones((src_img.shape[0], 
                              src_img.shape[1],1), dtype=np.float32)], axis=2)
    dst_img = np.concatenate([dst_img, np.ones((dst_img.shape[0],
                              dst_img.shape[1],1), dtype=np.float32)], axis=2)

    src_img, dst_img, src_lines, dst_lines, src_points, dst_points,\
        args.src_scale, args.dst_scale, _, _ = i_h.scale(src_img, dst_img, 
            src_lines, dst_lines, src_points, dst_points, args.scale_factor)

    # Stage one: Drawing points and perspective transform
    src_img, dst_img, src_points, dst_points, src_lines, dst_lines,\
        src_transform_matrix, dst_transform_matrix, stage_one_success =\
            stage_one(src_img, dst_img, src_points, dst_points, src_lines, 
                      dst_lines, args)
    
    # Stage two: Drawing lines and aaap morphing
    stage_two(src_img, dst_img, src_points, dst_points, src_lines, dst_lines, 
        src_transform_matrix, dst_transform_matrix, stage_one_success, args)
    
    # End
    print("Press ESC to quit program.")
    while cv2.waitKey(0) != 27:
        pass

    cv2.destroyAllWindows()
    i_h.vprint("Done.")
    exit()


if __name__ == "__main__":
    sys.exit(main())
