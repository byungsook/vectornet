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
tf.app.flags.DEFINE_integer('xy_size', 8,
                            """# Coordinates of two lines.""")
tf.app.flags.DEFINE_integer('min_length', 4,
                            """minimum length of a line.""")
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")

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
        self._func = partial(train_set, x_batch=self.x_batch, y_batch=self.y_batch)

    def batch(self):
        self._pool.map(self._func, range(FLAGS.batch_size))
        return self.x_batch, self.y_batch


def train_set(i, x_batch, y_batch):
    np.random.seed()
    while True:
        xy = np.random.randint(low=0, high=FLAGS.image_size, size=FLAGS.xy_size)
        if xy[0] - xy[2] + xy[1] - xy[3] < FLAGS.min_length:
            continue
        break

    SVG_LINE1 = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
        ) + SVG_LINE_TEMPLATE.format(
            id=0,
            x1=xy[0], y1=xy[1],
            x2=xy[2], y2=xy[3]
        ) + SVG_END_TEMPLATE

    SVG_MULTI_LINES = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
        )
    num_path = 2
    num_params_per_path = 4 # line
    # num_params_per_path = 8 # cubic bezier
    for path_id in range(num_path):
        start_p = num_params_per_path * path_id
        SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_LINE_TEMPLATE.format(
            id=path_id,
            x1=xy[start_p+0], y1=xy[start_p+1],
            x2=xy[start_p+2], y2=xy[start_p+3]
        )

    SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_END_TEMPLATE


    # save y png
    y_png = cairosvg.svg2png(bytestring=SVG_LINE1)
    y_img = Image.open(io.BytesIO(y_png))

    # load and normalize y to [0, 1]
    y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
    y = threshold(threshold(y, threshmin=0.5), threshmax=0.4, newval=1.0)
    y_batch[i,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])

    # select a random point on line1
    line_ids = np.nonzero(y)
    point_id = np.random.randint(len(line_ids[0]))
    px, py = line_ids[0][point_id], line_ids[1][point_id]

    # save x png
    x_png = cairosvg.svg2png(bytestring=SVG_MULTI_LINES)
    x_img = Image.open(io.BytesIO(x_png))

    # load and normalize y to [0, 0.1]
    x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
    x = threshold(threshold(x, threshmin=0.5), threshmax=0.4, newval=1.0/FLAGS.intensity_ratio)
    x[px, py] = 1.0 # 0.2 for debug
    x_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])


def batch():
    x_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    y_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    for i in xrange(FLAGS.batch_size):
        while True:
            xy = np.random.randint(low=0, high=FLAGS.image_size, size=FLAGS.xy_size)
            if xy[0] - xy[2] + xy[1] - xy[3] < FLAGS.min_length: 
                continue  
            break
        # print(xy.shape, xy.dtype)

        # # debug: one-pixel
        # xy = np.array([28, 43, 29, 43, 46, 22, 6, 32])

        SVG_LINE1 = SVG_START_TEMPLATE.format(
                width=FLAGS.image_size,
                height=FLAGS.image_size
            ) + SVG_LINE_TEMPLATE.format(
                id=0,
                x1=xy[0], y1=xy[1],
                x2=xy[2], y2=xy[3]
            ) + SVG_END_TEMPLATE

        SVG_TWO_LINES = SVG_START_TEMPLATE.format(
                width=FLAGS.image_size,
                height=FLAGS.image_size
            ) + SVG_LINE_TEMPLATE.format(
                id=0,
                x1=xy[0], y1=xy[1],
                x2=xy[2], y2=xy[3]
            ) + SVG_LINE_TEMPLATE.format(
                id=1,
                x1=xy[4], y1=xy[5],
                x2=xy[6], y2=xy[7]
            ) + SVG_END_TEMPLATE


        # save y png
        y_png = cairosvg.svg2png(bytestring=SVG_LINE1)
        y_img = Image.open(io.BytesIO(y_png))

        # load and normalize y to [0, 1]
        y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
        y = threshold(threshold(y, threshmin=0.5), threshmax=0.4, newval=1.0)
        y_batch[i,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])

        plt.imshow(y, cmap=plt.cm.gray)
        plt.show()

        # select a random point on line1
        line_ids = np.nonzero(y)
        # if len(line_ids) == 0:
        #     print(xy)
        point_id = np.random.randint(len(line_ids[0]))
        px, py = line_ids[0][point_id], line_ids[1][point_id]
        
        # save x png
        x_png = cairosvg.svg2png(bytestring=SVG_TWO_LINES)
        x_img = Image.open(io.BytesIO(x_png))
        
        # load and normalize y to [0, 0.1]
        x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
        x = threshold(threshold(x, threshmin=0.5), threshmax=0.4, newval=1.0/FLAGS.intensity_ratio)
        x[px, py] = 1.0 # 0.2 for debug
        
        # debug
        plt.imshow(x, cmap=plt.cm.gray)
        plt.show()

        x_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
                    
    return x_batch, y_batch


if __name__ == '__main__':
    # test
    batch()