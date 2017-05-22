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

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def svg2pdf(data_dir, dst_dir):
    cmap = plt.get_cmap('jet')

    file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
    count = 0
    with open(file_list_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if count >= 100: break

            file = line.rstrip()
            file_path = os.path.join(FLAGS.data_dir, file)
            dst_path = os.path.join(dst_dir, file)
            dst_path = dst_path.replace('svg_pre', 'pdf')
            print(file_path)

            with open(file_path, 'r') as sf:
                svg = sf.read()

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
                    w=64, h=64,
                    r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])

            cnorm  = colors.Normalize(vmin=0, vmax=num_path-1)
            cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

            end = 0
            # # ch1
            # stroke_color = ''' style="fill:rgb({r}, {g}, {b})" /'''
            # ch2
            stroke_color = ''' stroke="rgb({r}, {g}, {b})"'''
            for i in xrange(num_path):
                rgb = cscalarmap.to_rgba(i)
                r = int(rgb[0]*255)
                g = int(rgb[1]*255)
                b = int(rgb[2]*255)

                start = svg.find('path', end)
                # # ch1
                # end1 = svg.find('>', start)
                # end = svg.find('>', end1+1)
                # svg = svg[:end1] + stroke_color.format(r=r,g=g,b=b) + svg[end:]
                # # ch1
                # ch2
                end = svg.find('/>', start)
                svg = svg[:end] + stroke_color.format(r=r,g=g,b=b) + svg[end:]
                # ch2
                    
            cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=dst_path, scale=16)
            count += 1


    #  for root, _, files in os.walk(data_dir):
    #     for file in files:
    #         if not file.endswith('svg'): continue
            
    #         file_path = os.path.join(root, file)
    #         dst_path = os.path.join(dst_dir, file)
    #         dst_path = dst_path.replace('svg', 'png')

    #         print(file_path)
    #         with open(file_path, 'r') as sf:
    #             svg = sf.read()
    #             img = cairosvg.svg2png(bytestring=svg.encode('utf-8'),scale=1.0)
    #             img = Image.open(io.BytesIO(img))
    #             img = 1.0 - np.array(img)[:,:,3] / 255.0
    #             scipy.misc.imsave(dst_path, img)


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                default='data/chinese2',
                help='data directory',
                nargs='?') # optional arg.

    parser.add_argument('dst_dir',
                    default='paper/ch2',
                    help='destination directory',
                    nargs='?') # optional arg.
    
    parser.add_argument('file_list',
                    default='test.txt',
                    help='-',
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

    svg2pdf(FLAGS.data_dir, FLAGS.dst_dir)

    print('Done')
