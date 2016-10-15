import cv2 
import matlab.engine
import os, sys
import numpy as np
from image_morphing import morph


location = os.path.abspath(os.path.dirname(sys.argv[0]))
dataname = 'data.mat'


def init_matlab(path, src_name, dst_name):
    eng = matlab.engine.connect_matlab()
    eng.eval('clear',nargout=0)

    eng.workspace['mp_dataname'] = dataname
    eng.workspace['mp_path'] = path
    eng.workspace['mp_location'] = location
    eng.workspace['mp_src_name'] = src_name
    eng.workspace['mp_dst_name'] = dst_name

    eng.eval('cd(mp_location);')
    eng.eval('load([mp_path, mp_dataname]);', nargout=0)
    return eng


def draw_line(img, start, end):
    thickness = 1
    lineType = 8
    cv2.line(img, start, end, ( 0, 0, 255 ), thickness, lineType )


pressed = False
drag_start = (0,0)
def onMouse(event, x, y, flags, args):
    global pressed, drag_start
    img = args[0]
    lines = args[1]
    win_name = args[2]
    active_is_src_img = args[3]
    
    img_tmp = np.copy(img)

    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        drag_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and pressed:
        draw_line(img_tmp, drag_start,(x,y))
        cv2.imshow(win_name, img_tmp)
          
    elif event == cv2.EVENT_LBUTTONUP and pressed:
        draw_line(img, drag_start,(x,y))
        cv2.imshow(win_name, img)
        pressed = False
        lines.append([drag_start[0], drag_start[1], x, y])
    

def test():
    if len(sys.argv) != 4:
        print("Usage: test <path> <img_src> <img_dst>")
        exit()

    path = sys.argv[1]
    src_name = sys.argv[2]
    dst_name = sys.argv[3]

    src_img = cv2.imread(path + src_name)
    dst_img = cv2.imread(path + dst_name)

    if src_img is None:
        print ("Image 1 not readable or not found")
        exit()
    if dst_img is None:
        print ("Image 2 not readable or not found")
        exit()

    eng = init_matlab(path, src_name, dst_name)

    #print ("Draw rectangles with LMB to search for corresponding point.")
    #print ("Draw rectangles with RMB to only mark point.")
    #print ("Click i.e. draw very tiny rectangle to mark point directly.")
    #print ("Press Space to start morphing, ESC to quit")

    src_img_orig = np.copy(src_img)

    src_lines = []
    dst_lines = []

    cv2.namedWindow("src_img", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("dst_img", cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("src_img", onMouse, (src_img, src_lines, 'src_img', True))
    cv2.setMouseCallback("dst_img", onMouse, (dst_img, dst_lines, 'dst_img', False))
    cv2.imshow("src_img", src_img)
    cv2.imshow("dst_img", dst_img)
    cv2.resizeWindow("src_img", 640, 1024)
    cv2.resizeWindow("dst_img", 640, 1024)

    #src_l = eng.workspace['linesrc']
    #dst_l = eng.workspace['linedst']
    
    #y_max = src_img.shape[0]
    #for line in src_l:
    #    draw_line(dst_img, (int(line[0]), y_max - int(line[1])), (int(line[2]), y_max - int(line[3])))

    #for line in dst_l:
    #    draw_line(src_img, (int(line[0]), y_max - int(line[1])), (int(line[2]), y_max - int(line[3])))

    #cv2.imshow("src_img", src_img)
    #cv2.imshow("dst_img", dst_img)

    key = 0
    while key != 32:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            eng.quit()
            exit()

    cv2.destroyAllWindows()

    src_img = np.copy(src_img_orig)



    y_max = src_img.shape[0]
    for line in src_lines:
        line[1] = y_max - line[1]
        line[3] = y_max - line[3]
    for line in dst_lines:
        line[1] = y_max - line[1]
        line[3] = y_max - line[3]

    linesrc = matlab.double(src_lines)
    linedst = matlab.double(dst_lines)
    eng.workspace['linesrc'] = linedst
    eng.workspace['linedst'] = linesrc
    print eng.workspace['linesrc']
    print
    print eng.workspace['linedst']
    print

    eng.workspace['lineConstraintType'] = 2
    x, y, triangulation, quads = eng.eval('test(mp_path, mp_dst_name, mp_src_name, gridSize, linesrc, linedst, nSamplePerGrid, lineConstraintType, deformEnergyWeights)', nargout=4)

    points_old = []
    points_new = []
    max_x = dst_img.shape[1]
    max_y = dst_img.shape[0]


    for point in x:
        points_old.append((point[0], max_y - point[1]))

    for point in y:
        points_new.append((point[0], max_y - point[1]))


    src_img_morphed = morph(dst_img, src_img.shape, points_old, points_new, quads)
    cv2.namedWindow('src', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('src_morphed', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('dst', cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('dst_morphed', cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('overlay', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('src', src_img)
    cv2.imshow('src_morphed', src_img_morphed)
    cv2.imshow('dst', dst_img)
    print dst_img.shape
    print src_img_morphed.shape
    cv2.resizeWindow('src', 640, 480)
    cv2.resizeWindow('src_morphed', 640, 480)
    cv2.resizeWindow('dst', 640, 480)
    #overlay_img = (cv2.normalize(dst_img,dst_img, 0, 1, cv2.NORM_MINMAX)  +  cv2.normalize(src_img_morphed,src_img_morphed,0,1,cv2.NORM_MINMAX)) /2
    #overlay_img = np.zeros(src_img_morphed.shape,dtype=dst_img.dtype)
    #overlay_img[:,:,0] = dst_img[:,:,0] 
    #overlay_img[:,:,1] = src_img_morphed[:,:,1]
    #overlay_img[:,:,2] = src_img_morphed[:,:,2]
    #cv2.imshow('overlay', np.uint8(cv2.normalize(overlay_img, overlay_img, 0, 255, cv2.NORM_MINMAX)))
    cv2.resizeWindow('dst_morphed', 640, 480)
    cv2.imwrite('morphed.jpg', src_img_morphed)
    #cv2.resizeWindow('overlay', 640, 480)

    while cv2.waitKey(0) != 27:
        pass

    cv2.destroyAllWindows()
    eng.quit()
    exit()


if __name__ == "__main__":
    sys.exit(test())
