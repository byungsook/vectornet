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


def svg2png(data_dir, dst_dir):
     for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith('svg'): continue
            
            file_path = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, file)
            dst_path = dst_path.replace('svg', 'png')

            print(file_path)
            with open(file_path, 'r') as sf:
                svg = sf.read()
                img = cairosvg.svg2png(bytestring=svg.encode('utf-8'),scale=0.5)
                img = Image.open(io.BytesIO(img))
                img = 1.0 - np.array(img)[:,:,3] / 255.0
                scipy.misc.imsave(dst_path, img)


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                default='data/fidelity/preprocess', # 'data_tmp/gc_test',
                help='data directory',
                nargs='?') # optional arg.

    parser.add_argument('dst_dir',
                    default='data/fidelity/png_512', # 'data_tmp/gc_test',
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

    svg2png(FLAGS.data_dir, FLAGS.dst_dir)

    print('Done')
