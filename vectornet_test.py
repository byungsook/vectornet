# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
from os.path import basename
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cairosvg

import tensorflow as tf
from linenet.linenet_manager import LinenetManager
from beziernet.beziernet_manager import BeziernetManager


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'test/test',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'data/iter',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_lines', 10,
                           """maximum number of line to extract""")
tf.app.flags.DEFINE_integer('extract_iter', 2,
                           """iteration number for line extraction""")
tf.app.flags.DEFINE_float('epsilon', 0.02,
                           """epsilon for image comparision""")


SVG_START_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none" stroke-width="1">"""
SVG_CUBIC_BEZIER_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})"/>"""
SVG_END_TEMPLATE = """</g></svg>"""


def _imread(img_file_name, inv=False):
    """ Read, grayscale and normalize the image"""
    img = np.array(Image.open(img_file_name).convert('L')).astype(np.float) / 255.0
    if inv: 
        return 1.0 - img
    else: 
        return img
    

def draw_lines(lines, image_size, img_rec_name):
    SVG_MULTI_LINES = SVG_START_TEMPLATE.format(
            height=image_size[0],
            width=image_size[1]
        )
    
    np.random.seed(0) # to fix line colors
    for path_id, line in enumerate(lines):
        color = np.random.randint(low=0, high=256, size=3)
        SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_CUBIC_BEZIER_TEMPLATE.format(
            id=0,
            sx=line[0], sy=line[1],
            cx1=line[2], cy1=line[3],
            cx2=line[4], cy2=line[5],
            tx=line[6], ty=line[7],
            r=color[0], g=color[1], b=color[2]
        )
    SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_END_TEMPLATE

    cairosvg.svg2png(bytestring=SVG_MULTI_LINES, write_to=img_rec_name)
    return _imread(img_rec_name, inv=False), SVG_MULTI_LINES


def sample_pixel(img, img_rec):
    # select a pixel randomly
    img_line_ids = np.nonzero((img - img_rec) >= 0.5)

    img_pid = np.random.randint(len(img_line_ids[0]))
    return img_line_ids[0][img_pid], img_line_ids[1][img_pid]


def success_reconstruction(img, img_rec, epsilon):
    # check if the reconstructed sketch is similar to the original sketch
    err = (lambda img1, img2: np.sum((img1 - img2) ** 2) / float(img1.shape[0] * img1.shape[1]))(img, img_rec)
    print('%s: Err %.5f' % (datetime.now(), err))
    return err < epsilon


def sketch_simplify(img):
    """ simplify the input image """
    return img


def vectorize(linenet_manager, beziernet_manager, file_path):
    file_name = os.path.splitext(basename(file_path))[0]
    print('%s: %s, start vectorize.' % (datetime.now(), file_name))
    
    # read a image and normalize
    img = _imread(file_path, inv=True)
    
    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # paths for saving
    img_rec = np.empty(shape=img.shape)
    img_rec_name = os.path.join(FLAGS.test_dir, file_name + '_rec_%d.png')
    img_line_file_name = os.path.join(FLAGS.test_dir, file_name + '_rec_%d.svg')
    img_pdf_file_name = os.path.join(FLAGS.test_dir, file_name + '_rec_%d.pdf')
    img_extract_x_file_name = os.path.join(FLAGS.test_dir, file_name + '_extract_%d_%d_x.png')
    img_extract_y_file_name = os.path.join(FLAGS.test_dir, file_name + '_extract_%d_%d_y.png')

    # # debug
    # img_rec_name = 'data/test1/machine_rec.PNG'
    # img_rec = _imread(img_rec_name, inv=True)
    # plt.imshow(img_rec, cmap=plt.cm.gray)
    # plt.show()

    # vectorized lines
    lines = []

    # (optional)
    img_simple = sketch_simplify(img)
    for num_line in xrange(FLAGS.max_lines):
        start_time = time.time()                
        px, py = sample_pixel(img_simple, img_rec)
        duration = time.time() - start_time
        print('%s: line %d, sample pixel (%.3f sec)' % (datetime.now(), num_line, duration))

        # line extraction
        img_line = img_simple
        for i in xrange(FLAGS.extract_iter):
            start_time = time.time()
            img_x, img_line = linenet_manager.extract_line(img_line, px, py)
            Image.fromarray(np.uint8(img_x*255)).convert('L').save(img_extract_x_file_name % (num_line, i))
            Image.fromarray(np.uint8(img_line*255)).convert('L').save(img_extract_y_file_name % (num_line, i))
            duration = time.time() - start_time
            print('%s: line %d, extract line, iter %d (%.3f sec)' % (datetime.now(), num_line, i, duration))

        # line fitting
        start_time = time.time()                
        line = beziernet_manager.fit_line(img_line)
        lines.append(line)
        duration = time.time() - start_time
        print('%s: line %d, fit line, %s (%.3f sec)' % (datetime.now(), num_line, line, duration))

        # debug
        # xy = np.random.randint(low=0, high=min(img_simple.shape[0], img_simple.shape[1]), size=8)
        # line = [d for d in xy]
                        
        # update and save recontructed image
        img_rec, svg = draw_lines(lines, img_rec.shape, img_rec_name % num_line)
        with open(img_line_file_name % num_line, 'w') as f:
            f.write(svg)

        cairosvg.svg2pdf(bytestring=svg, write_to=img_pdf_file_name % num_line)

        # # debug
        # plt.imshow(img_rec, cmap=plt.cm.gray)
        # plt.show()
                
        if success_reconstruction(img_simple, img_rec, FLAGS.epsilon):
            break

    f.close()


def test():
    # create managers
    start_time = time.time()
    print('%s: Linenet manager loading...' % datetime.now())
    fixed_image_size = [48, 48]
    linenet_manager = LinenetManager(fixed_image_size)
    beziernet_manager = BeziernetManager(fixed_image_size)    
    duration = time.time() - start_time
    print('%s: manager loading (%.3f sec)' % (datetime.now(), duration))
    
    for root, _, files in os.walk(FLAGS.data_dir):
        for file in files:
            if not file.lower().endswith('png'):
                continue
            # elif file.lower().endswith('_rec.png'):
            #     continue
            
            file_path = os.path.join(root, file)
            start_time = time.time()
            vectorize(linenet_manager, beziernet_manager, file_path)
            duration = time.time() - start_time
            print('%s: %s processed (%.3f sec)' % (datetime.now(), file, duration))

    print('Done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('vectornet'):
        working_path = os.path.join(current_path, 'vectornet')
        os.chdir(working_path)
    
    # create test directory
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)

    # start
    test()


if __name__ == '__main__':
    tf.app.run()