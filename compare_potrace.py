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
import numpy as np
import scipy.misc
from subprocess import call

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import xml.etree.ElementTree as et

# flags
FLAGS = None


def compare_potrace():
    num_files = 0
    file_path_list = []        
    file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
    with open(file_list_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break

            file = line.rstrip()
            file_path = os.path.join(FLAGS.data_dir, file)
            file_path_list.append(file_path)
    # for root, _, files in os.walk('result/overlap_gco/ch1_'):
    #     for file in files:
    #         if not file.lower().endswith('svg'):
    #             continue
            
    #         file_name = file.split('_')[0]
    #         file_path = os.path.join(FLAGS.data_dir, file_name+'.svg_pre')
    #         file_path_list.append(file_path)

    # select test files
    num_total_test_files = len(file_path_list)
    FLAGS.num_test_files = min(num_total_test_files, FLAGS.num_test_files)
    # np.random.seed(0)
    # file_path_list_id = np.random.choice(num_total_test_files, FLAGS.num_test_files)
    # file_path_list.sort()
    file_path_list_id = xrange(FLAGS.num_test_files)

    acc_avg_total = 0.0
    for file_path_id in file_path_list_id:
        file_path = file_path_list[file_path_id]
        print(file_path)

        bmp_file_path = svgpre2bmp(file_path)
        svg_file_path = run_potrace(bmp_file_path)
        acc_avg = compute_accuracy(file_path, svg_file_path)
        acc_avg_total += acc_avg
        new_file_name = svg_file_path[:-4] + '_%.2f' % acc_avg + svg_file_path[-4:]
        os.rename(svg_file_path, new_file_name)

    acc_avg_total /= FLAGS.num_test_files
    print('acc_avg: %.3f' % acc_avg_total)
        

def svgpre2bmp(file_path):
    with open(file_path, 'r') as f:
        svg = f.read()
        num_path = svg.count('path d')
        if num_path == 0:
            # c2
            num_path = svg.count('path id')
            r = 0
            s = [1, 1] 
            t = [0, 0] 
        else:
            # c1
            r = 0
            s = [1, -1]
            t = [0, -900]
        svg = svg.format(
                w=FLAGS.image_width, h=FLAGS.image_height,
                r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity        
        s = 1.0 - s

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        bmp_file_path = os.path.join(FLAGS.dst_dir, file_name+'.bmp')
        scipy.misc.imsave(bmp_file_path, s)
    return bmp_file_path


def run_potrace(bmp_file_path):
    call(['data_tmp/potrace/potrace', 
          '-s',
        #   '-a 0', # corner threshold, 0: sharp, default: 1
          bmp_file_path])

    svg_file_path = bmp_file_path[:-3] + 'svg'
    with open(svg_file_path, 'r') as f:
        svg = f.read()
        num_paths = svg.count('path')

        cmap = plt.get_cmap('jet')    
        cnorm  = colors.Normalize(vmin=0, vmax=num_paths-1)
        cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

        end = 0
        stroke_color = ''' style="fill:rgb({r}, {g}, {b})"'''
        for i in xrange(num_paths):
            rgb = cscalarmap.to_rgba(i)
            r = int(rgb[0]*255)
            g = int(rgb[1]*255)
            b = int(rgb[2]*255)

            start = svg.find('path', end)
            end = svg.find('/>', start)
            svg = svg[:end] + stroke_color.format(r=r,g=g,b=b) + svg[end:]

    with open(svg_file_path, 'w') as f:
        f.write(svg)

    return svg_file_path


def compute_accuracy(file_path, svg_file_path):
    stroke_list = get_stroke_list(file_path)

    acc_id_list = []
    acc_list = []

    path_starts = []
    with open(svg_file_path, 'r') as f:
        svg = f.read()
        num_paths = svg.count('path')        

    for i in xrange(num_paths):
        svg_xml = et.fromstring(svg)
        stroke = svg_xml[1][i]
        for c in reversed(xrange(num_paths)):
            if svg_xml[1][c] != stroke:
                svg_xml[1].remove(svg_xml[1][c])
        svg_one_stroke = et.tostring(svg_xml, method='xml')
        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke_img.thumbnail((FLAGS.image_width, FLAGS.image_height), Image.ANTIALIAS)
        i_label_map = (np.array(stroke_img)[:,:,3] > 0)

        # # debug
        # plt.imshow(i_label_map, cmap=plt.cm.gray)
        # plt.show()

        accuracy_list = []
        for j, stroke in enumerate(stroke_list):
            intersect = np.sum(np.logical_and(i_label_map, stroke))
            union = np.sum(np.logical_or(i_label_map, stroke))
            accuracy = intersect / float(union)
            # print('compare with %d-th path, intersect: %d, union :%d, accuracy %.2f' % 
            #     (j, intersect, union, accuracy))
            accuracy_list.append(accuracy)

        id = np.argmax(accuracy_list)
        acc = np.amax(accuracy_list)
        # print('%d-th label, match to %d-th path, max: %.2f' % (i, id, acc))
        # consider only large label set
        # if acc > 0.1:
        acc_id_list.append(id)
        acc_list.append(acc)

    # print('avg: %.2f' % np.average(acc_list))
    return np.average(acc_list)


def get_stroke_list(file_path):
    stroke_list = []
    with open(file_path, 'r') as f:
        svg = f.read()
        r = 0
        s = [1, -1]
        t = [0, -900]

        svg = svg.format(
            w=FLAGS.image_width, h=FLAGS.image_height,
            r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])

        svg_xml = et.fromstring(svg)
        num_paths = len(svg_xml[0])

        for i in xrange(num_paths):
            svg_xml =et.fromstring(svg)
            stroke = svg_xml[0][i]
            for c in reversed(xrange(num_paths)):
                if svg_xml[0][c] != stroke:
                    svg_xml[0].remove(svg_xml[0][c])
            svg_one_stroke = et.tostring(svg_xml, method='xml')

            y_png = cairosvg.svg2png(bytestring=svg_one_stroke)
            y_img = Image.open(io.BytesIO(y_png))
            y = (np.array(y_img)[:,:,3] > 0)

            # # debug
            # y_img = np.array(y_img)[:,:,3].astype(np.float) / 255.0
            # plt.imshow(y_img, cmap=plt.cm.gray)
            # plt.show()

            stroke_list.append(y)

    return stroke_list


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                default='data/chinese1', # 'data_tmp/gc_test',
                help='data directory',
                nargs='?') # optional arg.
    parser.add_argument('dst_dir',
                    default='result/compare/chinese1', # 'data_tmp/gc_test',
                    help='destination directory',
                    nargs='?') # optional arg.
    parser.add_argument('file_list',
                    default='test.txt',
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('num_test_files',
                    default=100,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('image_width',
                    default=64,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('image_height',
                    default=64,
                    help='',
                    nargs='?') # optional arg.
    return parser.parse_args()


if __name__ == '__main__':
    # if release mode, change current path
    working_path = os.getcwd()
    if not working_path.endswith('vectornet'):
        working_path = os.path.join(working_path, 'vectornet')
        os.chdir(working_path)

    FLAGS = init_arg_parser()    
    if not os.path.exists(FLAGS.dst_dir):
        os.makedirs(FLAGS.dst_dir)

    compare_potrace()

    print('Done')