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
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_width', 128,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 128,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")
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
        self.num_examples_per_epoch = 1000
        self.num_epoch = 1

        if FLAGS.num_processors > FLAGS.batch_size:
            FLAGS.num_processors = FLAGS.batch_size

        if FLAGS.num_processors == 1:
            self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
        else:
            class MPManager(multiprocessing.managers.SyncManager):
                pass
            MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)

            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)
            
            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self._func = partial(train_set, x_batch=self.x_batch, y_batch=self.y_batch)


    def __del__(self):
        if FLAGS.num_processors > 1:
            self._pool.terminate() # or close
            self._pool.join()


    def batch(self):
        if FLAGS.num_processors == 1:
            for i in xrange(FLAGS.batch_size):
                train_set(i, self.x_batch, self.y_batch)
        else:
            self._pool.map(self._func, range(FLAGS.batch_size))

        return self.x_batch, self.y_batch



def train_set(batch_id, x_batch, y_batch):
    np.random.seed()
    
    svg = SVG_START_TEMPLATE.format(
                width=FLAGS.image_width,
                height=FLAGS.image_height
            )
    y = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.int)
    stroke_list = []
    for i in xrange(FLAGS.num_paths):
        LINE1 = _create_a_path(FLAGS.path_type, i)
        svg_one_stroke = SVG_START_TEMPLATE.format(
                width=FLAGS.image_width,
                height=FLAGS.image_height
            ) + LINE1 + SVG_END_TEMPLATE
        svg += LINE1

        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke.encode('utf-8'))
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke = (np.array(stroke_img)[:,:,3] > 0)

        # # debug
        # stroke_img = Image.open(io.BytesIO(stroke_png))
        # # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
        # plt.imshow(stroke_img)
        # plt.show()

        stroke_list.append(stroke)

    svg += SVG_END_TEMPLATE
    x_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    x_img = Image.open(io.BytesIO(x_png))
    x = np.array(x_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(x)

    for i in xrange(FLAGS.num_paths-1):
        for j in xrange(i+1, FLAGS.num_paths):
            intersect = np.logical_and(stroke_list[i], stroke_list[j])
            y = np.logical_or(intersect, y)
            # y += intersect # for multiple intersections

            # # debug
            # plt.figure()
            # plt.subplot(221)
            # plt.imshow(stroke_list[i], cmap=plt.cm.gray)
            # plt.subplot(222)
            # plt.imshow(stroke_list[j], cmap=plt.cm.gray)
            # plt.subplot(223)
            # plt.imshow(intersect, cmap=plt.cm.gray)
            # plt.subplot(224)
            # plt.imshow(y, cmap=plt.cm.gray)
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            # plt.show()

    # y = np.multiply(x, y)
    y = y.astype(np.float) * 1000
    
    # # debug
    # print('max intersection', np.amax(y))
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(x_img)
    # plt.subplot(122)
    # plt.imshow(y, cmap=plt.cm.gray)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()

    x_batch[batch_id,:,:,0] = x
    y_batch[batch_id,:,:,0] = y


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('ovnet'):
        working_path = os.path.join(current_path, 'vectornet/ovnet')
        os.chdir(working_path)

    # parameters 
    FLAGS.num_processors = 1

    batch_manager = BatchManager()
    x_batch, y_batch = batch_manager.batch()
    
    for i in xrange(FLAGS.batch_size):
        plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()

        # scipy.misc.imsave(current_path + '/inter_x%d.png' % i, np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]))
        # scipy.misc.imsave(current_path + '/inter_y%d.png' % i, np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]))

    print('Done')
