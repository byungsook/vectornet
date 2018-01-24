from __future__ import print_function

import os
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime
import imageio
from glob import glob
import shutil

def prepare_dirs_and_logger(config):
    # print(__file__)
    os.chdir(os.path.dirname(__file__))    

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # data path
    config.data_path = os.path.join(config.data_dir, config.dataset)

    # model path
    if config.is_train:
        model_name = os.path.join(config.archi, '{}_{}_{}'.format(
            config.dataset, get_time(), config.tag))
        config.model_dir = os.path.join(config.log_dir, model_name)    
    else:
        model_name = os.path.join('vec', '{}_{}_{}'.format(
            config.dataset, get_time(), config.tag))
        config.model_dir = os.path.join(config.log_dir, model_name)

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.ones([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)*255
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False, single=False):
    if not single:
        ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                                normalize=normalize, scale_each=scale_each)
    else:
        h, w = tensor.shape[0], tensor.shape[1]
        ndarr = np.zeros([h,w,3], dtype=np.uint8)
        ndarr[:,:] = tensor[:,:]
        
    im = Image.fromarray(ndarr)
    im.save(filename)

def convert_png2mp4(imgdir, filename, fps, delete_imgdir=False):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    try:
        writer = imageio.get_writer(filename, fps=fps)
    except Exception:
        imageio.plugins.ffmpeg.download()
        writer = imageio.get_writer(filename, fps=fps)

    imgs = sorted(glob("{}/*.png".format(imgdir)))
    # print(imgs)
    for img in imgs:
        im = imageio.imread(img)
        writer.append_data(im)
    
    writer.close()
    
    if delete_imgdir: shutil.rmtree(imgdir)
    
def rf(o, k, stride): # input size from output size
    return (o-1)*stride + k

def receptive_field_size(c, k, s):
    if c == 0:
        return rf(rf(1, k, 1), k, 1)
    else:
        rfs = receptive_field_size(c-1, k, s)
        print('%d: %d' % (c-1, rfs))
        return rf(rfs, k, s)

if __name__ == '__main__':
    c, k, s = 4, 3, 2
    rfs = receptive_field_size(c, k, s)
    print('c{}k{}s{} receptive field size'.format(c, k, s), rfs)

    c, k = 3, 3
    rfs = receptive_field_size(c, k, s)
    print('c{}k{}s{} receptive field size'.format(c, k, s), rfs)

    c, k = 5, 3
    rfs = receptive_field_size(c, k, s)
    print('c{}k{}s{} receptive field size'.format(c, k, s), rfs)

    c, k = 4, 4
    rfs = receptive_field_size(c, k, s)
    print('c{}k{}s{} receptive field size'.format(c, k, s), rfs)

    c, k = 3, 4
    rfs = receptive_field_size(c, k, s)
    print('c{}k{}s{} receptive field size'.format(c, k, s), rfs)

