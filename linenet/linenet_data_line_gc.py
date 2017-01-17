# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import io
from random import shuffle
import xml.etree.ElementTree as ET
import copy
import multiprocessing.managers
import multiprocessing.pool
from functools import partial
import scipy.misc

import numpy as np
import matplotlib.pyplot as plt

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('min_length', 10,
                            """minimum length of a line.""")
tf.app.flags.DEFINE_integer('num_paths', 5,
                            """# paths for batch generation""")
tf.app.flags.DEFINE_integer('path_type', 2,
                            """path type 0:line, 1:curve, 2:both""")
tf.app.flags.DEFINE_integer('max_stroke_width', 5,
                          """max stroke width""")


SVG_START_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none">"""
SVG_LINE_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" />"""
SVG_CUBIC_BEZIER_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" />"""
SVG_END_TEMPLATE = """</g></svg>"""


def _create_a_line(id, image_height, image_width, min_length):
    stroke_color = np.random.randint(240, size=3)
    stroke_width = np.random.rand() * FLAGS.max_stroke_width + 1
    while True:
        x = np.random.randint(low=0, high=image_width, size=2)
        y = np.random.randint(low=0, high=image_height, size=2)
        if x[0] - x[1] + y[0] - y[1] < min_length:
            continue
        break

    return SVG_LINE_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        x2=x[1], y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2], 
        sw=stroke_width
    )


def _create_a_cubic_bezier_curve(id, image_height, image_width, min_length):
    x = np.random.randint(low=0, high=image_width, size=4)
    y = np.random.randint(low=0, high=image_height, size=4)
    stroke_color = np.random.randint(240, size=3)
    stroke_width = np.random.rand() * FLAGS.max_stroke_width + 1

    return SVG_CUBIC_BEZIER_TEMPLATE.format(
        id=id,
        sx=x[0], sy=y[0],
        cx1=x[1], cy1=y[1],
        cx2=x[2], cy2=y[2],
        tx=x[3], ty=y[3],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2], 
        sw=stroke_width
    )


def _create_a_path(path_type, id):
    if path_type == 2:
        path_type = np.random.randint(2)

    path_selector = {
        0: _create_a_line,
        1: _create_a_cubic_bezier_curve
    }

    return path_selector[path_type](id, FLAGS.image_height, FLAGS.image_width, FLAGS.min_length)


class BatchManager(object):
    def __init__(self):
        pass

    def input(self):
        img, svg = generate()
        return img, svg, FLAGS.num_paths


def generate():
    np.random.seed()
    
    while True:
        svg = SVG_START_TEMPLATE.format(
                    width=FLAGS.image_width,
                    height=FLAGS.image_height
                )

        for i in xrange(FLAGS.num_paths):
            LINE1 = _create_a_path(FLAGS.path_type, i)
            svg += LINE1

        svg += SVG_END_TEMPLATE
        s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        
        if max_intensity == 0:
            continue
        else:
            s = s / max_intensity

        # # debug
        # plt.imshow(s_img)
        # plt.show()

        break

    return s, svg


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('linenet'):
        working_path = os.path.join(current_path, 'vectornet/linenet')
        os.chdir(working_path)

    # parameters
    tf.app.flags.DEFINE_integer('image_width', 64, '')
    tf.app.flags.DEFINE_integer('image_height', 64, '')

    batch_manager = BatchManager()
    img, svg, num_paths = batch_manager.input()
    
    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    print('Done')
