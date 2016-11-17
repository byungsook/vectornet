# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import argparse
from datetime import datetime

import cairosvg
from PIL import Image
import io
import matplotlib.pyplot as plt
import tarfile

import numpy as np
import scipy.stats
import scipy.misc


def preprocess_kanji(file_path):
    with open(file_path, 'r') as f:
        svg = ''
        while True:
            svg_line = f.readline()
            if svg_line.find('<svg') >= 0:
                id_width = svg_line.find('width')
                id_viewBox = svg_line.find('viewBox')
                svg = svg + svg_line[:id_width] + 'width="{w}" height="{h}" ' + svg_line[id_viewBox:]
                break
            else:
                svg = svg + svg_line

        # # optional: transform
        # svg = svg + '<g transform="scale({sx}, {sy}) translate({tx}, {ty})">\n'
        
        while True:
            svg_line = f.readline()
            # optional: stroke-width
            if svg_line.find('<g id="kvg:StrokePaths') >= 0:
                id_style = svg_line.find('stroke-width')
                svg = svg + svg_line[:id_style] + '">'
                continue
                
            if svg_line.find('<g id="kvg:StrokeNumbers') >= 0:
                # svg = svg + '</g>\n' # optional: transform
                svg = svg + '</svg>'
                break
            else:
                svg = svg + svg_line
        
    # # debug: test svg
    # img = cairosvg.svg2png(bytestring=svg.format(w=48, h=48, sx=1, sy=1, tx=0, ty=0))
    # img = Image.open(io.BytesIO(img))
    # img = np.array(img)[:,:,3].astype(np.float) / 255.0
    # # img = scipy.stats.threshold(img, threshmax=0.0001, newval=1.0)
    # img = 1.0 - img
    
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # save_path = os.path.join(FLAGS.dst_dir, os.path.splitext(os.path.basename(file_path))[0] + '_48.png')
    # scipy.misc.imsave(save_path, img)

    return svg


def preprocess_makemeahanzi(file_path):
    with open(file_path, 'r') as f:
        svg = f.readline()
        id_viewBox = svg.find('viewBox')
        svg = svg[:id_viewBox] + 'width="{w}" height="{h}" ' + svg[id_viewBox:]

        first_g = False
        while True:
            svg_line = f.readline()
            if svg_line.find('<g') >= 0:
                if first_g is False:
                    first_g = True
                else:
                    # svg = svg + '<g transform="rotate({r},512,512) translate({tx},{ty})">\n'
                    svg = svg + svg_line
                    break

        while True:
            svg_line = f.readline()
            if svg_line.find('<clipPath id=') >= 0:
                svg_line = f.readline()
                svg = svg + svg_line
            elif svg_line.find('</g>') >= 0:
                # svg = svg + '</g>\n'
                svg = svg + '</g>\n</svg>'
                break

    # # debug: test svg
    # img = cairosvg.svg2png(bytestring=svg.format(w=48, h=48))
    # img = Image.open(io.BytesIO(img))
    # img = np.array(img)[:,:,3].astype(np.float) / 255.0
    # img = 1.0 - img
    
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # save_path = os.path.join(FLAGS.dst_dir, os.path.splitext(os.path.basename(file_path))[0] + '_48.png')
    # scipy.misc.imsave(save_path, img)

    return svg


def preprocess_sketch(file_path):
    with open(file_path, 'r') as f:
        svg = f.readline()
        id_width = svg.find('width')
        id_xmlns = svg.find('xmlns', id_width)
        svg_size = 'width="{w}" height="{h}" viewBox="0 0 640 480" '
        svg = svg[:id_width] + svg_size + svg[id_xmlns:]
        
        while True:
            svg_line = f.readline()
            if svg_line.find('<g') >= 0:
                svg = svg + svg_line
                break

        # gather normal paths and remove thick white stroke
        while True:
            svg_line = f.readline()
            if not svg_line:
                break
            elif svg_line.find('<g') >= 0:
                svg = svg + '</svg>'
                break

            # filter thick white strokes
            id_white_stroke = svg_line.find('#fff')
            if id_white_stroke == -1:
                svg = svg + svg_line


    # # debug
    # img = cairosvg.svg2png(bytestring=svg.format(w=48, h=48))
    # img = Image.open(io.BytesIO(img))                
    # img = np.array(img)[:,:,3].astype(np.float) / 255.0
    # img = scipy.stats.threshold(img, threshmax=0.0001, newval=1.0)
    # img = 1.0 - img
    
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # save_path = os.path.join(FLAGS.dst_dir, os.path.splitext(os.path.basename(file_path))[0] + '_48.png')
    # scipy.misc.imsave(save_path, img)

    return svg


def preprocess(run_id):
    if run_id == 0:
        data_dir = 'data_tmp/svg_test' 
    elif run_id == 1:    
        data_dir = 'linenet/data/chinese/makemeahanzi/svgs'
    elif run_id == 2:
        data_dir = 'linenet/data/chinese/kanjivg-20160426-all/kanji'

    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith('svg'):
                continue

            file_path = os.path.join(root, file)

            # check validity of svg file
            try:
                cairosvg.svg2png(url=file_path)
            except Exception as e:
                continue
            
            # parsing..
            if run_id == 0:
                svg_pre = preprocess_sketch(file_path) 
            elif run_id == 1:
                svg_pre = preprocess_makemeahanzi(file_path)
            elif run_id == 2:
                svg_pre = preprocess_kanji(file_path)

            # write preprocessed svg
            write_path = os.path.join(FLAGS.dst_dir, file[:-3] + 'svg_pre')
            with open(write_path, 'w') as f:
                f.write(svg_pre)

    # compress
    with tarfile.open(FLAGS.dst_tar, "w:gz") as tar:
        tar.add(FLAGS.dst_dir, arcname=os.path.basename(FLAGS.dst_dir))


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dst_dir',
                    default='linenet/data/chinese2', # 'data_tmp/gc_test',
                    help='destination directory',
                    nargs='?') # optional arg.
    parser.add_argument('dst_tar',
                    default='linenet/data/chinese2.tar.gz', # 'data_tmp/gc_test',
                    help='destination tar file',
                    nargs='?') # optional arg.
    return parser.parse_args()


if __name__ == '__main__':
    # if release mode, change current path
    working_path = os.getcwd()
    if not working_path.endswith('vectornet'):
        working_path = os.path.join(working_path, 'vectornet')
        os.chdir(working_path)
    
    # flags
    FLAGS = init_arg_parser()
    
    if not os.path.exists(FLAGS.dst_dir):
        os.makedirs(FLAGS.dst_dir)

    # run [0-2]
    preprocess(2)

    print('Done')
