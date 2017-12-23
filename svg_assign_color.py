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

    for root, _, files in os.walk(FLAGS.data_dir):
        for file in files:
            if not file.lower().endswith('svg'):
                continue

            file_path = os.path.join(FLAGS.data_dir, file)
            dst_path = os.path.join(dst_dir, file)
            print(file_path)

            with open(file_path, 'r') as sf:
                svg = sf.read()
            
            file_pdf_path = file_path[:-3] + 'pdf'
            cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=file_pdf_path)
            file_png_path = file_path[:-3] + 'png'
            cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=file_png_path)

            num_path = svg.count('stroke:#')
            cnorm = colors.Normalize(vmin=0, vmax=num_path-1)
            cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

            from random import shuffle
            c_list = [i for i in range(num_path)]
            shuffle(c_list)
            c_end = -1
            for c in c_list:
                rgb = cscalarmap.to_rgba(c)
                r = int(rgb[0]*255)
                g = int(rgb[1]*255)
                b = int(rgb[2]*255)

                c_start = svg.find('stroke:#', c_end+1)
                if c_start == -1:
                    break

                c_end = svg.find(';', c_start+1)
                # svg = svg[:c_start] + 'stroke:#%02x%02x%02x' % (r,g,b) + svg[c_end:]
                svg = svg[:c_start] + 'stroke:#000000' + svg[c_end:]
            
            with open(dst_path, 'w') as f:
                f.write(svg)

            dst_pdf_path = dst_path[:-3] + 'pdf'
            cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=dst_pdf_path)
            dst_png_path = dst_path[:-3] + 'png'
            cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=dst_png_path)


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                default='data/fidelity/256',
                help='data directory',
                nargs='?') # optional arg.

    parser.add_argument('dst_dir',
                    default='data/fidelity/black',
                    help='destination directory',
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
