# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from scipy.misc import imread
from scipy.stats import threshold
import cairosvg
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool
import multiprocessing.managers
from functools import partial
from PIL import Image
import io

import tensorflow as tf

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 48, # 48-24-12-6
                            """Image Size.""")
tf.app.flags.DEFINE_integer('min_length', 4,
                            """minimum length of a line.""")
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")
tf.app.flags.DEFINE_integer('num_path', 4,
                            """# paths for batch generation""")
tf.app.flags.DEFINE_integer('path_type', 2,
                            """path type 0: line, 1: curve, 2: both""")

SVG_START_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none" stroke="black" stroke-width="1">"""
SVG_LINE_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}"/>"""
SVG_CUBIC_BEZIER_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}"/>"""
SVG_END_TEMPLATE = """</g></svg>"""


class BatchManager(object):
    """
    Batch Manager using multiprocessing
    """
    def __init__(self):
        class MPManager(multiprocessing.managers.BaseManager):
            pass
        MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)

        self._mpmanager = MPManager()
        self._mpmanager.start()
        self._pool = Pool(processes=8)
        self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
        self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
        self.x_no_p_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
        self._func = partial(train_set, x_batch=self.x_batch, y_batch=self.y_batch, x_no_p_batch=self.x_no_p_batch)

    def __del__(self):
        self._pool.terminate() # or close
        self._pool.join()

    def batch(self):
        self._pool.map(self._func, range(FLAGS.batch_size))
        return self.x_batch, self.y_batch, self.x_no_p_batch


def _create_a_line():
    while True:
        xy = np.random.randint(low=0, high=FLAGS.image_size, size=4)
        if xy[0] - xy[2] + xy[1] - xy[3] < FLAGS.min_length:
            continue
        break

    return SVG_LINE_TEMPLATE.format(
        id=0,
        x1=xy[0], y1=xy[1],
        x2=xy[2], y2=xy[3]
    )


def _create_a_cubic_bezier_curve():
    xy = np.random.randint(low=0, high=FLAGS.image_size, size=8)
    return SVG_CUBIC_BEZIER_TEMPLATE.format(
        id=0,
        sx=xy[0], sy=xy[1],
        cx1=xy[2], cy1=xy[3],
        cx2=xy[4], cy2=xy[5],
        tx=xy[6], ty=xy[7]
    )


def _create_a_path(path_type):
    if path_type == 2:
        path_type = np.random.randint(2)

    path_selector = {
        0: _create_a_line,
        1: _create_a_cubic_bezier_curve
    }
    return path_selector[path_type]()


def train_set(i, x_batch, y_batch, x_no_p_batch):
    np.random.seed()
    LINE1 = _create_a_path(FLAGS.path_type)
    SVG_LINE1 = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
        ) + LINE1 + SVG_END_TEMPLATE

    SVG_MULTI_LINES = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
        )
    SVG_MULTI_LINES = SVG_MULTI_LINES + LINE1
    for path_id in range(1, FLAGS.num_path):
        SVG_MULTI_LINES = SVG_MULTI_LINES + _create_a_path(FLAGS.path_type)
    SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_END_TEMPLATE


    # save y png
    y_png = cairosvg.svg2png(bytestring=SVG_LINE1)
    y_img = Image.open(io.BytesIO(y_png))
    # y_img.save('log/png/y_img%d.png' % i)
    # y_img = Image.open('log/png/%d_y_img%d.png' % (step, i))

    # load and normalize y to [0, 1]
    y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
    # y = threshold(threshold(y, threshmin=0.5), threshmax=0.4, newval=1.0)
    y_batch[i,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])

    # select a random point on line1
    line_ids = np.nonzero(y > 0.4)
    point_id = np.random.randint(len(line_ids[0]))
    px, py = line_ids[0][point_id], line_ids[1][point_id]

    # save x png
    x_png = cairosvg.svg2png(bytestring=SVG_MULTI_LINES)
    x_img = Image.open(io.BytesIO(x_png))
    # x_img.save('log/png/x_img%d.png' % i)
    # x_img = Image.open('log/ratio_test/png/%d_x_img%d.png' % (step, i))

    # load and normalize y to [0, 0.1]
    x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
    x_no_p_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])

    # x = threshold(threshold(x, threshmin=0.5), threshmax=0.4, newval=1.0/FLAGS.intensity_ratio)
    x = x / FLAGS.intensity_ratio
    x[px, py] = 1.0 # 0.2 for debug
    x_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])


def batch():
    x_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    y_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    for i in xrange(FLAGS.batch_size):
        np.random.seed()
        LINE1 = _create_a_path(FLAGS.path_type)
        SVG_LINE1 = SVG_START_TEMPLATE.format(
                width=FLAGS.image_size,
                height=FLAGS.image_size
            ) + LINE1 + SVG_END_TEMPLATE

        SVG_MULTI_LINES = SVG_START_TEMPLATE.format(
                width=FLAGS.image_size,
                height=FLAGS.image_size
            )
        SVG_MULTI_LINES = SVG_MULTI_LINES + LINE1
        for path_id in range(1, FLAGS.num_path):
            SVG_MULTI_LINES = SVG_MULTI_LINES + _create_a_path(FLAGS.path_type)
        SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_END_TEMPLATE


        # save y png
        y_png = cairosvg.svg2png(bytestring=SVG_LINE1)
        y_img = Image.open(io.BytesIO(y_png))

        # load and normalize y to [0, 1]
        y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
        # y = threshold(threshold(y, threshmin=0.5), threshmax=0.4, newval=1.0)
        y_batch[i,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])

        plt.imshow(y, cmap=plt.cm.gray)
        plt.show()

        # select a random point on line1
        line_ids = np.nonzero(y > 0.4)
        # if len(line_ids) == 0:
        #     print(xy)
        point_id = np.random.randint(len(line_ids[0]))
        px, py = line_ids[0][point_id], line_ids[1][point_id]
        
        # save x png
        x_png = cairosvg.svg2png(bytestring=SVG_MULTI_LINES)
        x_img = Image.open(io.BytesIO(x_png))
        
        # load and normalize y to [0, 0.1]
        x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
        #x = threshold(threshold(x, threshmin=0.5), threshmax=0.4, newval=1.0/FLAGS.intensity_ratio)
        x = x / FLAGS.intensity_ratio
        x[px, py] = 1.0 # 0.2 for debug
        
        # debug
        plt.imshow(x, cmap=plt.cm.gray)
        plt.show()

        x_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
                    
    return x_batch, y_batch


if __name__ == '__main__':
    # test
    batch()