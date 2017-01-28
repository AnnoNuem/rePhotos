import matlab.engine
import numpy as np 


def init_matlab():
    
    eng = matlab.engine.connect_matlab()
    eng.eval('clear', nargout=0)

    eng.eval('init_f_m', nargout=0)

    return eng


def get_template_location(img, tmp, eng):

    img = img.tolist()
    tmp = tmp.tolist()
    eng.workspace['img'] = img
    eng.workspace['template'] = tmp
    
    trans_mat = eng.eval('get_template_location(img, template)', nargout=1)

    return np.array(trans_mat)

def close_matlab(eng):
    eng.quit()


