import cv2 
import matlab.engine
import os, sys
import numpy as np
from image_helpers import scale
from image_morphing import morph


location = os.path.abspath(os.path.dirname(sys.argv[0]))
dataname = 'data.mat'


def init_matlab():
    eng = matlab.engine.connect_matlab()
    eng.eval('clear',nargout=0)

    eng.workspace['mp_dataname'] = dataname
    eng.workspace['mp_location'] = location

    eng.eval('cd(mp_location);')
    eng.eval('load([mp_dataname]);', nargout=0)
    return eng


def draw_line(img, start, end, color):
    thickness = int((img.shape[0] + img.shape[1]) / 1000  )
    lineType = 8
    cv2.line(img, start, end, color, thickness, lineType )


pressed = False
drag_start = (0,0)
def onMouse(event, x, y, flags, args):
    global pressed, drag_start
    img = args[0]
    lines = args[1]
    win_name = args[2]
    color = args[3]
    active_is_src_img = args[4]
    
    img_tmp = np.copy(img)

    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        drag_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and pressed:
        draw_line(img_tmp, drag_start,(x,y), color)
        cv2.imshow(win_name, img_tmp)
          
    elif event == cv2.EVENT_LBUTTONUP and pressed:
        draw_line(img, drag_start,(x,y), color)
        cv2.imshow(win_name, img)
        pressed = False
        lines.append([drag_start[0], drag_start[1], x, y])
    

def test():
    if len(sys.argv) != 3:
        print("Usage: test <img_src> <img_dst>")
        exit()
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

    eng = init_matlab()

    src_img_orig = np.copy(src_img)
    dst_img_orig = np.copy(dst_img)

    src_lines = []
    dst_lines = []

    cv2.namedWindow("src_img", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("dst_img", cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("src_img", onMouse, (src_img, src_lines, 'src_img', (255,255,0), True))
    cv2.setMouseCallback("dst_img", onMouse, (dst_img, dst_lines, 'dst_img', (0,0,255), False))
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
            eng.quit()
            exit()
    cv2.destroyAllWindows()

    #src_img = np.copy(src_img_orig)
    #dst_img = np.copy(dst_img_orig)

    # scale images, add small value to later crop everything which is zero
    src_img_alpha = np.ones((src_img.shape[0], src_img.shape[1], 4), np.float32)
    src_img_alpha[:, :, 0:3] = np.float32(src_img[:,:,0:3])
    dst_img_alpha = np.ones((dst_img.shape[0], dst_img.shape[1], 4), np.float32)
    dst_img_alpha[:, :, 0:3] = np.float32(dst_img[:,:,0:3])
    src_img_alpha, dst_img_alpha, src_lines, dst_lines = scale(src_img_alpha, dst_img_alpha, src_lines, dst_lines)

    y_max = src_img_alpha.shape[0]
    for line in src_lines:
        line[1] = y_max - line[1]
        line[3] = y_max - line[3]
    for line in dst_lines:
        line[1] = y_max - line[1]
        line[3] = y_max - line[3]

    # compute grid and grid deformation
    linesrc = matlab.double(src_lines)
    linedst = matlab.double(dst_lines)
    eng.workspace['linesrc'] = linedst
    eng.workspace['linedst'] = linesrc
    eng.workspace['width'] = float(src_img_alpha.shape[1])
    eng.workspace['height'] = float(src_img_alpha.shape[0])
    eng.workspace['lineConstraintType'] = 2
    x, y, triangulation, quads = eng.eval('test(gridSize, linesrc, linedst, nSamplePerGrid, \
        lineConstraintType, deformEnergyWeights, width, height)', nargout=4)

    # transform point coordinates from matlab to numpy
    points_old = []
    points_new = []
    max_x = dst_img_alpha.shape[1]
    max_y = dst_img_alpha.shape[0]
    for point in x:
        points_old.append((point[0], max_y - point[1]))
    for point in y:
        points_new.append((point[0], max_y - point[1]))

    # morph image
    (src_img_morphed, dst_img_cropped, src_img_cropped) = morph(dst_img_alpha, src_img_alpha, points_old, points_new, quads)

    # postprocess
    src_img_morphed = np.uint8(src_img_morphed[:, :, 0:3])
    dst_img_cropped = np.uint8(dst_img_cropped[:, :, 0:3])
    src_img_cropped = np.uint8(src_img_cropped[:, :, 0:3])
    src_img_alpha = np.uint8(src_img_alpha[:, :, 0:3])
    dst_img_alpha = np.uint8(dst_img_alpha[:, :, 0:3])

    overlay_morphed = cv2.addWeighted(dst_img_cropped, 0.5, src_img_morphed, 0.5, 0)
    overlay_orig = cv2.addWeighted(src_img_alpha, 0.5, dst_img_alpha, 0.5, 0)
    
    # display
    #cv2.namedWindow('src', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('src_morphed', cv2.WINDOW_KEEPRATIO)
    #cv2.namedWindow('dst', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('overlay', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('overlay', cv2.addWeighted(dst_img_cropped, 0.5, src_img_morphed, 0.5, 0))
    cv2.resizeWindow('overlay', 640, 480)
    #cv2.namedWindow('overlay_orig', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow('src', src_img)
    cv2.imshow('src_morphed', src_img_morphed)
    #cv2.imshow('dst', dst_img)
    #cv2.resizeWindow('src', 640, 480)
    cv2.resizeWindow('src_morphed', 640, 480)
    #cv2.resizeWindow('dst', 640, 480)
    

    #cv2.imshow('overlay_orig', cv2.addWeighted(dst_img_cropped, 0.5, src_img_cropped, 0.5, 0))
    #cv2.resizeWindow('dst_morphed', 640, 480)
    #cv2.resizeWindow('overlay_orig', 640, 480)

    # write2disk
    filenname_prefix = 'results/' + (src_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + \
        '_' + (dst_name.rsplit('/',1)[-1]).rsplit('.',1)[0] + '__'
    cv2.imwrite(filenname_prefix + 'src.jpg', dst_img_alpha)
    cv2.imwrite(filenname_prefix + 'dst.jpg', src_img_alpha)
    cv2.imwrite(filenname_prefix + 'src_morphed.jpg', src_img_morphed)
    cv2.imwrite(filenname_prefix + 'dst_cropped.jpg', dst_img_cropped)
    cv2.imwrite(filenname_prefix + 'src_cropped.jpg', src_img_cropped)
    cv2.imwrite(filenname_prefix + 'overlay_morphed.jpg', overlay_morphed)
    cv2.imwrite(filenname_prefix + 'overlay_orig.jpg', overlay_orig)

    while cv2.waitKey(0) != 27:
        pass

    cv2.destroyAllWindows()
    eng.quit()
    exit()


if __name__ == "__main__":
    sys.exit(test())
