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


def split_dataset():
    file_list = []
    for root, _, files in os.walk(FLAGS.dst_dir):
        for file in files:
            if not file.lower().endswith('svg_pre'):
                continue

            file_list.append(file)

    num_files = len(file_list)
    ids = np.random.permutation(num_files)
    train_id = int(num_files * 0.9)
    with open(os.path.join(FLAGS.dst_dir,'train.txt'), 'w') as f: 
        for id in ids[:train_id]:
            f.write(file_list[id] + '\n')
    with open(os.path.join(FLAGS.dst_dir,'test.txt'), 'w') as f: 
        for id in ids[train_id:]:
            f.write(file_list[id] + '\n')


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

        # optional: transform
        # sy=-1, ty=-109, sy=1, ty=0
        svg = svg + '<g transform="rotate({r},54,54) scale({sx},{sy}) translate({tx},{ty})">\n'
        
        while True:
            svg_line = f.readline()
            # # optional: stroke-width
            # if svg_line.find('<g id="kvg:StrokePaths') >= 0:
            #     id_style = svg_line.find('stroke-width')
            #     svg = svg + svg_line[:id_style] + '">'
            #     continue

            if svg_line.find('<g id="kvg:StrokeNumbers') >= 0:
                svg = svg + '</g>\n' # optional: transform
                svg = svg + '</svg>'
                break
            else:
                svg = svg + svg_line
        
    # # debug: test svg
    # img = cairosvg.svg2png(bytestring=svg.format(w=128, h=128, r=30, sx=1, sy=1, tx=0, ty=0))
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
                    # sy=-1, ty=-900, sy=1, ty=124 (1024-900)
                    svg = svg + '<g transform="rotate({r},512,512) scale({sx},{sy}) translate({tx},{ty})">\n'
                    break

        while True:
            svg_line = f.readline()
            if svg_line.find('<clipPath id=') >= 0:
                # read and add a path
                svg_line = f.readline()
                svg = svg + svg_line
            elif svg_line.find('</g>') >= 0:
                svg = svg + '</g>\n</svg>'
                break

    # # debug: test svg
    # img = cairosvg.svg2png(bytestring=svg.format(
    #                        w=96, h=96, r=20, sx=1.5, sy=2, tx=100, ty=224))
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
    # img = cairosvg.svg2png(bytestring=svg.format(w=128, h=96))
    # img = Image.open(io.BytesIO(img))                
    # img = np.array(img)[:,:,3].astype(np.float) / 255.0
    # # img = scipy.stats.threshold(img, threshmax=0.0001, newval=1.0)
    # # img = 1.0 - img
    
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # save_path = os.path.join(FLAGS.dst_dir, os.path.splitext(os.path.basename(file_path))[0] + '_48.png')
    # scipy.misc.imsave(save_path, img)

    return svg


def preprocess_fidelity(file_path):
    with open(file_path, 'r') as f:
        svg = f.read()

    # debug
    img = cairosvg.svg2png(bytestring=svg)
    img = Image.open(io.BytesIO(img))
    # ratio = 512 / max(img.size[0], img.size[0])
    # width = int(ratio * img.size[0])
    # height = int(ratio * img.size[1])
    # img = img.resize((width, height))
    img = np.array(img)[:,:,3].astype(np.float) / 255.0
    # img = scipy.stats.threshold(img, threshmax=0.0001, newval=1.0)
    # img = 1.0 - img
    
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    save_path = os.path.join(FLAGS.dst_dir, os.path.splitext(os.path.basename(file_path))[0] + '.png')
    scipy.misc.imsave(save_path, img)


def preprocess(run_id):
    if run_id == 0:
        data_dir = 'linenet/data/sketches'
    elif run_id == 1:
        data_dir = 'linenet/data/chinese/makemeahanzi/svgs'
    elif run_id == 2:
        data_dir = 'linenet/data/chinese/kanjivg-20160426-all/kanji'
    elif run_id == 3:
        data_dir = 'data_tmp/fidelity/output/svg'

    if run_id == 0:
        valid_file_list_name = 'checked.txt'
        for root, _, files in os.walk(data_dir):
            if not valid_file_list_name in files:
                continue

            valid_file_list_path = os.path.join(root, valid_file_list_name)
            with open(valid_file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    file = line.rstrip('\n') + '.svg'
                    file_path = os.path.join(root, file)

                    # check validity of svg file
                    try:
                        cairosvg.svg2png(url=file_path)
                    except Exception as e:
                        continue

                    svg_pre = preprocess_sketch(file_path)

                    # write preprocessed svg
                    write_path = os.path.join(FLAGS.dst_dir, file[:-3] + 'svg_pre')
                    with open(write_path, 'w') as wf:
                        wf.write(svg_pre)

    else:
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
                if run_id == 1:
                    svg_pre = preprocess_makemeahanzi(file_path)
                elif run_id == 2:
                    svg_pre = preprocess_kanji(file_path)
                elif run_id == 3:
                    svg_pre = preprocess_fidelity(file_path)


                # write preprocessed svg
                write_path = os.path.join(FLAGS.dst_dir, file[:-3] + 'svg_pre')
                with open(write_path, 'w') as f:
                    f.write(svg_pre)

    # split train/test dataset
    split_dataset()

    # compress
    with tarfile.open(FLAGS.dst_tar, "w:gz") as tar:
        tar.add(FLAGS.dst_dir, arcname=os.path.basename(FLAGS.dst_dir))


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('process_num',
                    default=0,
                    help='process number',
                    nargs='?') 
    parser.add_argument('dst_dir',
                    default='linenet/data/sketch', # 'data_tmp/gc_test',
                    help='destination directory',
                    nargs='?') # optional arg.
    parser.add_argument('dst_tar',
                    default='linenet/data/sketch.tar.gz', # 'data_tmp/gc_test',
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

    # run [0-3]
    preprocess(FLAGS.process_num)

    print('Done')
