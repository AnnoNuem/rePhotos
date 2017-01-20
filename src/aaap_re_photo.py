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
from time import sleep
from image_aaap_main import aaap_morph
from image_sac import getPointFromPoint
from image_sac import getPointFromRectangle
from image_perspective_alignment import perspective_align


line_file_name_end = "line_file.txt"
point_file_name_end = "point_file.txt"
    
drag_start = (0,0)
point_stage = True
number_of_points = 0
def onMouse(event, x, y, flags, (img, img_orig, lines, points, win_name, color)):
    global drag_start, point_stage, number_of_points

    img_tmp = np.copy(img)
    if point_stage:
        if event == cv2.EVENT_LBUTTONUP and len(points) < 4:
            points.append((x,y))
            number_of_points = number_of_points + 1
            i_h.draw_circle(img, (x,y), color)
            cv2.imshow(win_name, img)
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) < 4:
            drag_start = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON\
             and len(points) < 4:
            i_h.draw_rectangle(img_tmp, drag_start, (x,y), color)
            cv2.imshow(win_name, img_tmp)
        elif event == cv2.EVENT_RBUTTONUP and len(points) < 4:
            point = getPointFromRectangle(img, drag_start, (x,y))
            i_h.draw_circle(img, point, color)
            cv2.imshow(win_name, img)
            points.append(point)
            number_of_points = number_of_points + 1
        elif event == cv2.EVENT_MBUTTONUP and len(points) > 0:
            del points[-1]
            img[:] = img_orig[:]
            number_of_points -= 1
            for point in points:
                i_h.draw_circle(img, point, color)
            cv2.imshow(win_name, img)

    elif not point_stage:
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            drag_start = getPointFromPoint(img, (x, y))
        elif event == cv2.EVENT_MOUSEMOVE and (flags==cv2.EVENT_FLAG_LBUTTON\
             or flags==cv2.EVENT_FLAG_RBUTTON):
            i_h.draw_line(img_tmp, drag_start,(x,y), color, -1)
            cv2.imshow(win_name, img_tmp)
        elif event == cv2.EVENT_LBUTTONUP:
            distance = math.sqrt((x-drag_start[0])**2 + (y-drag_start[1])**2)
            if distance < 2:
                lines.append([drag_start[0], drag_start[1]-1, drag_start[0], 
                             drag_start[1]+1])
                i_h.draw_line(img, (drag_start[0]-1, drag_start[1]),
                              (drag_start[0]+1, drag_start[1]), color, 
                              len(lines))
                lines.append([drag_start[0]-1, drag_start[1], drag_start[0]+1, 
                             drag_start[1]])
                i_h.draw_line(img, (drag_start[0], drag_start[1]-1),
                              (drag_start[0], drag_start[1]+1), color, 
                              len(lines))
            else:
                lines.append([drag_start[0], drag_start[1], x, y])
                i_h.draw_line(img, drag_start,(x,y), color, len(lines))
            cv2.imshow(win_name, img)
        elif event == cv2.EVENT_RBUTTONUP:
            drag_end = getPointFromPoint(img, (x, y))
            line = i_l.get_line(drag_start, drag_end, img_orig)
            lines.append(line)
            i_h.draw_line(img,(line[0], line[1]), (line[2], line[3]), (0,255,0), 
                          len(lines))
            cv2.imshow(win_name, img)
        elif event == cv2.EVENT_MBUTTONUP and len(lines) > 0:
            del lines[-1]
            # np.copy creates new array, local img points to new array, 
            # main img still points to old img with lines
            img[:] = img_orig[:]
            i = 1
            for line in lines:
                i_h.draw_line(img, (line[0], line[1]), (line[2], line[3]), color, i)
                i+=1
            for point in points:
                i_h.draw_circle(img, point, color)
            cv2.imshow(win_name, img)


def test():
    global point_stage, number_of_points

    # argparsing
    parser = argparse.ArgumentParser()

    parser.add_argument("src_name", help="source image")
    parser.add_argument("dst_name", help="destination image")
    parser.add_argument("-dg", "--draw_grid", help="draw aaap warp grid on\
                        result images", action="store_true")
    parser.add_argument("-l", "--line_file", help="read lines from and save lines\
                        to line file", action="store_true")
    parser.add_argument("-p", "--point_file", help="read points from and save points\
                        to point file", action="store_true")
    parser.add_argument("-sf", "--show_frame", help="do not crop the resulting\
                        images but show a frame where images would be cropped",
                        action="store_true")
    parser.add_argument("-sui", "--show_user_input", help="Show user drawn points\
                        and lines in result images", action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-w", "--write", help="write result images to files in\
                        resultsfolder", action="store_true")
    args = parser.parse_args()

    # Init
    if args.verbose:
        i_h.set_verbose(True)

    src_img = cv2.imread(args.src_name)
    dst_img = cv2.imread(args.dst_name)
    if src_img is None:
        print("Image 1 not readable or not found")
        exit()
    if dst_img is None:
        print("Image 2 not readable or not found")
        exit()
    src_img_orig = np.copy(src_img)
    dst_img_orig = np.copy(dst_img)
    if args.line_file:
        line_file_name = args.src_name.rsplit('.', 1)[0] + "_" + line_file_name_end
        src_lines, dst_lines = i_io.read_lines(line_file_name)
    else:
        src_lines = []
        dst_lines = []

    # Stage One: Drawing Points:
    #
    #
    if args.point_file:
        point_file_name = args.src_name.rsplit('.', 1)[0] + "_" + point_file_name_end
        src_points, dst_points = i_io.read_points(point_file_name)
        number_of_points = len(src_points) + len(dst_points)
        for point in src_points:
            i_h.draw_circle(src_img, point, (255,255,0))
        for point in dst_points:
            i_h.draw_circle(dst_img, point, (0,0 ,255))
    else:
        src_points = []
        dst_points = []

    print("First Stage: Draw four points for initial perspective transform.\
           \nDrawing less than four points omits the first stage.\
           \nLMB: Draw point.\
           \nRMB: Draw rectangle. Best point in rectangle is computed.\
           \nMMB: Delete last line in active window.\
           \nSpace: Go to second stage.\
           \nESC: Quit program.\n")

    cv2.namedWindow("src_img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dst_img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("src_img", onMouse, (src_img, src_img_orig, src_lines, 
                         src_points, 'src_img', (255,255,0)))
    cv2.setMouseCallback("dst_img", onMouse, (dst_img, dst_img_orig, dst_lines, 
                         dst_points, 'dst_img', (0,0,255)))
    cv2.imshow("src_img", src_img)
    cv2.imshow("dst_img", dst_img)
    cv2.resizeWindow("src_img", 640, 1024)
    cv2.resizeWindow("dst_img", 640, 1024)

    # wait till user has drawn all points 
    key = 0
    while key != 32 and point_stage:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            exit()
    
    # Stage Two: Drawing Lines:
    #
    #
    point_stage = False
    print("Second Stage: Draw lines for fine grain aaap warping.\
          \nDrawing no line omits the second stage.\
          \nLMB: Draw Line\
          \nMMB: Delete last line in active Window\
          \nRMB: Drawing line with RMB finds nearest line\
          \nSpace: Start morphing\nEsc: Quit program")

    if args.line_file:
        i = 0
        for line in src_lines:
            i_h.draw_line(src_img, (line[0], line[1]), (line[2], line[3]), (255,255,0), i)
            i += 1
        i = 0
        for line in dst_lines:
            i_h.draw_line(dst_img, (line[0], line[1]), (line[2], line[3]), (0,0 ,255), i)
            i += 1
        cv2.imshow("src_img", src_img)
        cv2.imshow("dst_img", dst_img)

    # wait till user has drawn all lines
    key = 0
    while key != 32 :
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()

    if args.line_file:
        i_io.write_lines(src_lines, dst_lines, line_file_name)

    if args.point_file:
        i_io.write_points(src_points, dst_points, point_file_name)

    if args.show_user_input:
        src_img = np.float32(src_img)
        dst_img = np.float32(dst_img)
    else:
        src_img = np.float32(np.copy(src_img_orig))
        dst_img = np.float32(np.copy(dst_img_orig))

    # Image warping
    #
    #
    # scale images
    i_h.vprint("Scaling...")
    scale_factor = 4
    # add alpha channel for cropping
    src_img = np.concatenate([src_img, np.ones((src_img.shape[0], src_img.shape[1],1))], axis=2)
    dst_img = np.concatenate([dst_img, np.ones((dst_img.shape[0], dst_img.shape[1],1))], axis=2)

    src_img, dst_img, src_lines, dst_lines, src_points, dst_points, x_max, y_max = \
        i_h.scale(src_img, dst_img, src_lines, dst_lines, src_points, dst_points, scale_factor)


    # write2disk
    if args.write:
        if not os.path.exists('results'):
            os.makedirs('results')
        filenname_prefix = 'resutls/' + (args.src_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + \
            '_' + (args.dst_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + '__'
        cv2.imwrite(filenname_prefix + 'src_original.jpg', src_img)
        cv2.imwrite(filenname_prefix + 'dst_original.jpg', dst_img)

    # perspective alignment
    if number_of_points == 8:
        i_h.vprint("Perspective Alignment...")

        src_img, dst_img, src_points, dst_points, src_lines, dst_lines = perspective_align(src_img, dst_img, src_points, dst_points, src_lines, dst_lines, alpha=0.5)

        cv2.namedWindow('src_transformed', cv2.WINDOW_NORMAL)
        cv2.imshow('src_transformed', np.uint8(src_img))
        cv2.resizeWindow('src_transformed', 640, 480)

        cv2.namedWindow('dst_transformed', cv2.WINDOW_NORMAL)
        cv2.imshow('dst_transformed', np.uint8(dst_img))
        cv2.resizeWindow('dst_transformed', 640, 480)

        cv2.namedWindow('overlay_transformed', cv2.WINDOW_NORMAL)
        transform_overlay = np.uint8(cv2.addWeighted(src_img, 0.5, dst_img, 0.5, 0))
        cv2.imshow('overlay_transformed', transform_overlay)
        cv2.resizeWindow('overlay_transformed', 640, 480)

        # write2disk
        if args.write:
            if not os.path.exists('results'):
                os.makedirs('results')
            filenname_prefix = 'results/' + (args.src_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + \
                '_' + (args.dst_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + '__'
            cv2.imwrite(filenname_prefix + 'src_perspective_transform.jpg', src_img)
            cv2.imwrite(filenname_prefix + 'dst_perspective_transform.jpg', dst_img)
            cv2.imwrite(filenname_prefix + 'overlay_perspective_transform.jpg', transform_overlay)

        print("Perspective transform done.\
              \nPress SPACE to continue to aaap warping. ESC to exit.")

        key = 0
        while key != 32 :
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit()
    else:
        i_h.vprint("Not enough points for perspective transform. Skipping")
    cv2.destroyAllWindows()

    # aaap warping perspective transformed images
    if len(src_lines) > 0:    
        # add each point from perspective transform as two short orthagonal lines
        # to keep these points fixed during warping
        for point in src_points:
            src_lines.append([point[0], point[1], point[0]+1, point[1]])
            src_lines.append([point[0], point[1], point[0], point[1]+1])
        for point in dst_points:
            dst_lines.append([point[0], point[1], point[0]+1, point[1]])
            dst_lines.append([point[0], point[1], point[0], point[1]+1])

        # morph
        src_img_morphed, src_img_cropped, dst_img_cropped = aaap_morph(src_img, 
            dst_img, src_lines, dst_lines, line_constraint_type=2, grid_size=20, 
            scale_factor=scale_factor, show_frame=args.show_frame, 
            draw_grid_f=args.draw_grid)

        # compute overlay
        overlay_morphed = cv2.addWeighted(dst_img_cropped, 0.5, src_img_morphed, 0.5, 0)
        overlay_orig = cv2.addWeighted(dst_img_cropped, 0.5, src_img_cropped, 0.5, 0)
        
        # display
        cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
        cv2.imshow('overlay', overlay_morphed)
        cv2.resizeWindow('overlay', 640, 480)
        cv2.namedWindow('overlay_orig', cv2.WINDOW_NORMAL)
        cv2.imshow('overlay_orig', overlay_orig)
        cv2.resizeWindow('overlay_orig', 640, 480)
        cv2.namedWindow('src_morphed', cv2.WINDOW_NORMAL)
        cv2.imshow('src_morphed', src_img_morphed)
        cv2.resizeWindow('src_morphed', 640, 480)
        
        # write2disk
        if args.write:
            if not os.path.exists('results'):
                os.makedirs('results')
            if number_of_points == 8:
                filenname_prefix = 'results/' + (args.src_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + \
                    '_' + (args.dst_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + '_aaap_with_perspective__'
            else:
                filenname_prefix = 'results/' + (args.src_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + \
                    '_' + (args.dst_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + '_aaap_without_perspective__'
            cv2.imwrite(filenname_prefix + 'src_morphed.jpg', src_img_morphed)
            cv2.imwrite(filenname_prefix + 'dst_cropped.jpg', dst_img_cropped)
            cv2.imwrite(filenname_prefix + 'overlay_aaap.jpg', overlay_morphed)
    else:
        print("No lines for aaap warping. Skipping")
    
    print("Press ESC to quit program.")
    while cv2.waitKey(0) != 27:
        pass

    cv2.destroyAllWindows()
    i_h.vprint("Done.")
    exit()


if __name__ == "__main__":
    sys.exit(test())
