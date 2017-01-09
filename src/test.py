import cv2 
import numpy as np
import sys
from image_aaap_main import aaap_morph
import image_lines as i_l
from image_sac import getPointFromPoint
import json


use_line_file= True
line_file_name_end = "line_file.txt"

def read_lines(filename):
    try:
        f = open(filename, "r")
    except IOError:
        print("Could not open line file.")
        return [], []

    with f:
        [src_lines, dst_lines] = json.load(f)

    return src_lines, dst_lines


def write_lines(src_lines, dst_lines, filename):
    try:
        f = open(filename, "w")
    except IOError:
        print("Could not save lines to file.")

    with f:
        json.dump([src_lines, dst_lines], f)
    

def draw_line(img, start, end, color, l_number):
    thickness = int((img.shape[0] + img.shape[1]) / 900  ) + 1
    lineType = 8
    cv2.line(img, start, end, color, thickness, lineType )
    if l_number > 0:
        text_size = float(thickness)/2 
        cv2.putText(img, str(l_number), end, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, text_size, (0,255,0), thickness)


drag_start = (0,0)
def onMouse(event, x, y, flags, (img, img_orig, lines, win_name, color)):
    global drag_start
    
    img_tmp = np.copy(img)

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)

    if event == cv2.EVENT_RBUTTONDOWN:
        drag_start = getPointFromPoint(img, (x, y))

    elif event == cv2.EVENT_MOUSEMOVE and (flags==cv2.EVENT_FLAG_LBUTTON or flags==cv2.EVENT_FLAG_RBUTTON):
        draw_line(img_tmp, drag_start,(x,y), color, -1)
        cv2.imshow(win_name, img_tmp)
          
    elif event == cv2.EVENT_LBUTTONUP:
        lines.append([drag_start[0], drag_start[1], x, y])
        draw_line(img, drag_start,(x,y), color, len(lines))
        cv2.imshow(win_name, img)

    elif event == cv2.EVENT_RBUTTONUP:
        #line = i_l.get_line(drag_start, (x,y), img_orig, i_l.PST)
        #lines.append(line)
        #draw_line(img,(line[0], line[1]), (line[2], line[3]), color, len(lines))
        drag_end = getPointFromPoint(img, (x, y))
        line = i_l.get_line(drag_start, drag_end, img_orig, i_l.STAT_CANNY)
        lines.append(line)
        draw_line(img,(line[0], line[1]), (line[2], line[3]), (0,255,0), len(lines))
        cv2.imshow(win_name, img)

    elif event == cv2.EVENT_MBUTTONUP and len(lines) > 0:
        del lines[-1]
        # np.copy creates new array, local img points to new array, main img still points to old img with lines
        img[:] = img_orig[:]
        i = 1
        for line in lines:
            draw_line(img, (line[0], line[1]), (line[2], line[3]), color, i)
            i+=1
        cv2.imshow(win_name, img)


def test():
    if len(sys.argv) != 3:
        print("Usage: test <img_src> <img_dst>")
        exit()
    print("LMB: Draw Line\nMMB: Delete last line in active Window\nRMB: Drawing line with RMB finds nearest line\nSpace: Start morphing\nEsc: Quit program")
    src_name = sys.argv[1]
    dst_name = sys.argv[2]
    src_img = cv2.imread(src_name)
    dst_img = cv2.imread(dst_name)

    if src_img is None:
        print ("Image 1 not readable or not found")
        exit()
    if dst_img is None:
        print ("Image 2 not readable or not found")
        exit()

    src_img_orig = np.copy(src_img)
    dst_img_orig = np.copy(dst_img)

    if use_line_file:
        line_file_name = src_name.rsplit('.', 1)[0] + "_" + line_file_name_end
        src_lines, dst_lines = read_lines(line_file_name)
        i = 0
        for line in src_lines:
            draw_line(src_img, (line[0], line[1]), (line[2], line[3]), (255,255,0), i)
            i += 1
        i = 0
        for line in dst_lines:
            draw_line(dst_img, (line[0], line[1]), (line[2], line[3]), (0,0 ,255), i)
            i += 1
    else:
        src_lines = []
        dst_lines = []

    cv2.namedWindow("src_img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("dst_img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("src_img", onMouse, (src_img, src_img_orig, src_lines, 'src_img', (255,255,0)))
    cv2.setMouseCallback("dst_img", onMouse, (dst_img, dst_img_orig, dst_lines, 'dst_img', (0,0,255)))
    cv2.imshow("src_img", src_img)
    cv2.imshow("dst_img", dst_img)
    cv2.resizeWindow("src_img", 640, 1024)
    cv2.resizeWindow("dst_img", 640, 1024)

    # wait till user has drawn all lines
    key = 0
    while key != 32:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()

    if use_line_file:
        write_lines(src_lines, dst_lines, line_file_name)

    # morph
    src_img_morphed, dst_img_cropped, src_img_cropped, src_img_morphed_grid, src_img_grid = aaap_morph(src_img_orig, dst_img_orig, src_lines, dst_lines, line_constraint_type=2, grid_size=10, scale_factor=4)

    # compute overlay
    print dst_img_cropped.shape
    print src_img_morphed.shape
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
    cv2.namedWindow('src_grid', cv2.WINDOW_NORMAL)
    cv2.imshow('src_grid', src_img_grid)
    cv2.resizeWindow('src_grid', 640, 480)
    cv2.namedWindow('src_morphed_grid', cv2.WINDOW_NORMAL)
    cv2.imshow('src_morphed_grid', src_img_morphed_grid)
    cv2.resizeWindow('src_morphed_grid', 640, 480)
    
    # write2disk
    filenname_prefix = 'results/' + (src_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + \
        '_' + (dst_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + '__'
    #cv2.imwrite(filenname_prefix + 'src.jpg', dst_img)
    #cv2.imwrite(filenname_prefix + 'dst.jpg', src_img)
    cv2.imwrite(filenname_prefix + 'src_morphed.jpg', src_img_morphed)
    cv2.imwrite(filenname_prefix + 'dst_cropped.jpg', dst_img_cropped)
    #cv2.imwrite(filenname_prefix + 'src_cropped.jpg', src_img_cropped)
    #cv2.imwrite(filenname_prefix + 'overlay_morphed.jpg', overlay_morphed)
    cv2.imwrite(filenname_prefix + 'morphed_grid.jpg', src_img_morphed_grid)
    cv2.imwrite(filenname_prefix + 'grid.jpg', src_img_grid)
    #cv2.imwrite(filenname_prefix + 'overlay_orig.jpg', overlay_orig)

    while cv2.waitKey(0) != 27:
        pass

    cv2.destroyAllWindows()
    exit()


if __name__ == "__main__":
    sys.exit(test())
