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
from scipy import ndimage
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
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")
tf.app.flags.DEFINE_integer('image_size', 48, # 48-24-12-6
                            """Image Size.""")
tf.app.flags.DEFINE_integer('min_length', 4,
                            """minimum length of a line.""")
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")
tf.app.flags.DEFINE_integer('num_path', 2,
                            """# paths for batch generation""")
tf.app.flags.DEFINE_integer('path_type', 0,
                            """path type 0: line, 1: curve, 2: both""")
tf.app.flags.DEFINE_boolean('noise_on', False,
                            """noise on/off""")
tf.app.flags.DEFINE_integer('noise_rot_deg', 3,
                            """rotation degree for noise generation""")
tf.app.flags.DEFINE_integer('noise_trans_pix', 2,
                            """translation pixel for noise generation""")
tf.app.flags.DEFINE_integer('noise_duplicate_min', 2,
                            """min # duplicates for noise generation""")
tf.app.flags.DEFINE_integer('noise_duplicate_max', 6,
                            """max # duplicates for noise generation""")
tf.app.flags.DEFINE_float('noise_intensity', 30,
                          """unifor noise intensity""")
tf.app.flags.DEFINE_boolean('use_two_channels', False,
                            """use two channels for input""")
tf.app.flags.DEFINE_float('prob_background', 0.0,
                          """probability for selecting background""")


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
        self._pool = Pool(processes=FLAGS.num_processors)
        input_depth = 2 if FLAGS.use_two_channels else 1
        self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, input_depth], dtype=np.float)
        self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
        self.x_no_p_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
        self.p_batch = self._mpmanager.np_empty([FLAGS.batch_size, 2], dtype=np.int)
        self._func = partial(train_set, x_batch=self.x_batch, y_batch=self.y_batch, 
                             x_no_p_batch=self.x_no_p_batch, p_batch=self.p_batch)

    def __del__(self):
        self._pool.terminate() # or close
        self._pool.join()

    def batch(self):
        self._pool.map(self._func, range(FLAGS.batch_size))
        return self.x_batch, self.y_batch, self.x_no_p_batch, self.p_batch


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


def _slur_image(img):
    # img = face(gray=True)
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # gaussian blur
    gauss_denoised = ndimage.gaussian_filter(img, 0.1)
    # plt.imshow(gauss_denoised, cmap=plt.cm.gray)
    # plt.show()

    # duplicate
    num_duplicates = np.random.randint(low=FLAGS.noise_duplicate_min, high=FLAGS.noise_duplicate_max+1)
    blend = np.zeros(gauss_denoised.shape)
    for i in xrange(num_duplicates):
        # rotate
        rnd_offset = np.random.rand(1) * 2.0 - 1.0
        rotated_face = ndimage.rotate(gauss_denoised, rnd_offset * FLAGS.noise_rot_deg, reshape=False)
        
        # translate
        rnd_offset = np.random.rand(2) * 2.0 - 1.0
        shifted_face = ndimage.shift(rotated_face, rnd_offset * FLAGS.noise_trans_pix)
        
        # blend duplicates
        weight_min = 0.75
        weight = np.random.rand() * (1.0 - weight_min) + weight_min
        blend = blend + weight * shifted_face

    # add noise
    noisy = blend + FLAGS.noise_intensity * np.random.randn(*blend.shape)
    noisy = np.clip(noisy, a_min=0.0, a_max=255.0) / 255.0
    
    # # debug
    # plt.imshow(noisy, cmap=plt.cm.gray)
    # plt.show()
    
    return noisy


def batch_for_pbmap_test(seed):
    # center = FLAGS.image_size*0.5
    # px_list = [center-p for p in xrange(-5, 6)]
    # py_list = list(px_list)
    # num_pixels = len(px_list) ** 2
    num_pixels = FLAGS.image_size * FLAGS.image_size

    x_no_p_batch = np.empty([num_pixels, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    x_batch = np.empty([num_pixels, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    

    np.random.seed(seed)
    SVG_MULTI_LINES = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
        )
    for path_id in range(0, FLAGS.num_path):
        SVG_MULTI_LINES = SVG_MULTI_LINES + _create_a_path(FLAGS.path_type)
    SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_END_TEMPLATE

    
    x_png = cairosvg.svg2png(bytestring=SVG_MULTI_LINES)
    x_img = Image.open(io.BytesIO(x_png))
    if FLAGS.noise_on:
        x = _slur_image(np.array(x_img)[:,:,3])
    else:
        x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
    x_no_p = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
    x_norm = x / FLAGS.intensity_ratio

    i = 0
    for px in xrange(FLAGS.image_size):
        for py in xrange(FLAGS.image_size):
    # for px in px_list:
    #     for py in py_list:
            x_no_p_batch[i,:,:] = x_no_p
        
            # plt.imshow(x, cmap=plt.cm.gray)
            # plt.show()

            tmp = x_norm[px, py]
            x_norm[px, py] = 1.0
            x_batch[i,:,:] = np.reshape(x_norm, [FLAGS.image_size, FLAGS.image_size, 1])
            x_norm[px, py] = tmp

            # plt.imshow(x, cmap=plt.cm.gray)
            # plt.show()

            i = i + 1

    return num_pixels, x_batch, x_no_p_batch


def batch_for_intersection_test():
    center = [FLAGS.image_size*0.5] * 2
    test_range = 5
    num_lines = 4

    x_batch = np.empty([test_range*2*num_lines+1, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    y_batch = np.empty([test_range*2*num_lines+1, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    x_no_p_batch = np.empty([test_range*2*num_lines+1, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    p_batch = np.empty([test_range*2*num_lines+1, 2], dtype=np.int)

    # / \ - |
    xy1 = [FLAGS.image_size*0.25, FLAGS.image_size*0.25, FLAGS.image_size*0.75, FLAGS.image_size*0.75]
    xy2 = [FLAGS.image_size*0.25, FLAGS.image_size*0.75, FLAGS.image_size*0.75, FLAGS.image_size*0.25]
    xy3 = [FLAGS.image_size*0.25, FLAGS.image_size*0.50, FLAGS.image_size*0.75, FLAGS.image_size*0.50]
    xy4 = [FLAGS.image_size*0.50, FLAGS.image_size*0.25, FLAGS.image_size*0.50, FLAGS.image_size*0.75]

    LINE1 = SVG_LINE_TEMPLATE.format(
        id=1,
        x1=xy1[0], y1=xy1[1],
        x2=xy1[2], y2=xy1[3]
    )

    LINE2 = SVG_LINE_TEMPLATE.format(
        id=2,
        x1=xy2[0], y1=xy2[1],
        x2=xy2[2], y2=xy2[3]
    )

    LINE3 = SVG_LINE_TEMPLATE.format(
        id=3,
        x1=xy3[0], y1=xy3[1],
        x2=xy3[2], y2=xy3[3]
    )

    LINE4 = SVG_LINE_TEMPLATE.format(
        id=4,
        x1=xy4[0], y1=xy4[1],
        x2=xy4[2], y2=xy4[3]
    )

    SVG_LINE1 = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
    ) + LINE1 + SVG_END_TEMPLATE

    SVG_LINE2 = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
    ) + LINE2 + SVG_END_TEMPLATE

    SVG_LINE3 = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
    ) + LINE3 + SVG_END_TEMPLATE

    SVG_LINE4 = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
    ) + LINE4 + SVG_END_TEMPLATE

    SVG_LINES = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
    ) + LINE1 + LINE2 + LINE3 + LINE4 + SVG_END_TEMPLATE

    y1_png = cairosvg.svg2png(bytestring=SVG_LINE1)
    y1_img = Image.open(io.BytesIO(y1_png))
    y1 = np.array(y1_img)[:,:,3].astype(np.float) / 255.0

    y2_png = cairosvg.svg2png(bytestring=SVG_LINE2)
    y2_img = Image.open(io.BytesIO(y2_png))
    y2 = np.array(y2_img)[:,:,3].astype(np.float) / 255.0

    y3_png = cairosvg.svg2png(bytestring=SVG_LINE3)
    y3_img = Image.open(io.BytesIO(y3_png))
    y3 = np.array(y3_img)[:,:,3].astype(np.float) / 255.0

    y4_png = cairosvg.svg2png(bytestring=SVG_LINE4)
    y4_img = Image.open(io.BytesIO(y4_png))
    y4 = np.array(y4_img)[:,:,3].astype(np.float) / 255.0

    x_png = cairosvg.svg2png(bytestring=SVG_LINES)
    x_img = Image.open(io.BytesIO(x_png))
    x_no_p = np.array(x_img)[:,:,3].astype(np.float) / 255.0
    x = x_no_p / FLAGS.intensity_ratio
    
    j = 0
    for r in xrange(-test_range, test_range+1):
        y_batch[j,:,:] = np.reshape(y1, [FLAGS.image_size, FLAGS.image_size, 1])
        x_no_p_batch[j,:,:] = np.reshape(x_no_p, [FLAGS.image_size, FLAGS.image_size, 1])
        p_batch[j] = [center[0]+r, center[1]+r]
        tmp = x[p_batch[j,0],p_batch[j,1]]
        x[p_batch[j,0],p_batch[j,1]] = 1.0
        x_batch[j,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
        x[p_batch[j,0],p_batch[j,1]] = tmp

        j = j + 1
        if r != 0:
            y_batch[j,:,:] = np.reshape(y2, [FLAGS.image_size, FLAGS.image_size, 1])
            x_no_p_batch[j,:,:] = np.reshape(x_no_p, [FLAGS.image_size, FLAGS.image_size, 1])
            p_batch[j] = [center[0]+r, center[1]-r-1]
            tmp = x[p_batch[j,0],p_batch[j,1]]
            x[p_batch[j,0],p_batch[j,1]] = 1.0
            x_batch[j,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
            x[p_batch[j,0],p_batch[j,1]] = tmp
            j = j + 1

            y_batch[j,:,:] = np.reshape(y3, [FLAGS.image_size, FLAGS.image_size, 1])
            x_no_p_batch[j,:,:] = np.reshape(x_no_p, [FLAGS.image_size, FLAGS.image_size, 1])
            p_batch[j] = [center[0], center[1]+r]
            tmp = x[p_batch[j,0],p_batch[j,1]]
            x[p_batch[j,0],p_batch[j,1]] = 1.0
            x_batch[j,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
            x[p_batch[j,0],p_batch[j,1]] = tmp
            j = j + 1

            y_batch[j,:,:] = np.reshape(y4, [FLAGS.image_size, FLAGS.image_size, 1])
            x_no_p_batch[j,:,:] = np.reshape(x_no_p, [FLAGS.image_size, FLAGS.image_size, 1])
            p_batch[j] = [center[0]+r, center[1]]
            tmp = x[p_batch[j,0],p_batch[j,1]]
            x[p_batch[j,0],p_batch[j,1]] = 1.0
            x_batch[j,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
            x[p_batch[j,0],p_batch[j,1]] = tmp
            j = j + 1
    
    return x_batch, y_batch, x_no_p_batch, p_batch


def new_x_from_y_with_p(x_batch, y_batch, p_batch):
    """ generate new x_batch from y_bath with p_batch """
    for i in xrange(FLAGS.batch_size):
        x = np.reshape(y_batch[i,:,:], [FLAGS.image_size, FLAGS.image_size])
        x = x / FLAGS.intensity_ratio
        x[p_batch[i,0],p_batch[i,1]] = 1.0
        # plt.imshow(x, cmap=plt.cm.gray)
        # plt.show()
        x_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])


def train_set(i, x_batch, y_batch, x_no_p_batch, p_batch):
    np.random.seed()
    while True:
        LINE1 = _create_a_path(FLAGS.path_type)
        SVG_LINE1 = SVG_START_TEMPLATE.format(
                width=FLAGS.image_size,
                height=FLAGS.image_size
            ) + LINE1 + SVG_END_TEMPLATE

        y_png = cairosvg.svg2png(bytestring=SVG_LINE1)
        y_img = Image.open(io.BytesIO(y_png))
        y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
        y_batch[i,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])
        
        line_ids = np.nonzero(y >= 0.5)
        if len(line_ids[0]) > 0:
            break

    SVG_MULTI_LINES = SVG_START_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size
        )
    SVG_MULTI_LINES = SVG_MULTI_LINES + LINE1
    for path_id in range(1, FLAGS.num_path):
        SVG_MULTI_LINES = SVG_MULTI_LINES + _create_a_path(FLAGS.path_type)
    SVG_MULTI_LINES = SVG_MULTI_LINES + SVG_END_TEMPLATE


    point_id = np.random.randint(len(line_ids[0]))
    px, py = line_ids[0][point_id], line_ids[1][point_id]

    # save x png
    x_png = cairosvg.svg2png(bytestring=SVG_MULTI_LINES)
    x_img = Image.open(io.BytesIO(x_png))
    # x_img.save('log/png/x_img%d.png' % i)
    # x_img = Image.open('log/ratio_test/png/%d_x_img%d.png' % (step, i))

    if FLAGS.use_two_channels:
        x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
        x_no_p_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1]) 
        x_batch[i,:,:,0] = x

        x_point = np.zeros(x.shape)
        x_point[px, py] = 1.0
        x_batch[i,:,:,1] = x_point
    else:
        # load and normalize y to [0, 0.1]
        if FLAGS.noise_on:
            x = _slur_image(np.array(x_img)[:,:,3])
        else:
            x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
        x_no_p_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])

        # x = threshold(threshold(x, threshmin=0.5), threshmax=0.4, newval=1.0/FLAGS.intensity_ratio)
        x = x / FLAGS.intensity_ratio
        x[px, py] = 1.0 # 0.2 for debug
        x_batch[i,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])

        # plt.imshow(x, cmap=plt.cm.gray)
        # plt.show()

    p_batch[i,:] = [px, py]


def batch(check_result=False):
    input_depth = 2 if FLAGS.use_two_channels else 1
    x_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, input_depth], dtype=np.float)
    y_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    x_no_p_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    p_batch = np.empty([FLAGS.batch_size, 2], dtype=np.int)
    for i in xrange(FLAGS.batch_size):
        train_set(i, x_batch, y_batch, x_no_p_batch, p_batch)

    if check_result:
        if FLAGS.use_two_channels:
            t = np.concatenate((x_batch, np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1])), axis=3)
            plt.imshow(t[0,:,:,:], cmap=plt.cm.gray)
            plt.show()
        else:
            x = np.reshape(x_batch[0,:,:], [FLAGS.image_size, FLAGS.image_size])
            plt.imshow(x, cmap=plt.cm.gray)
            plt.show()

    return x_batch, y_batch, x_no_p_batch, p_batch


if __name__ == '__main__':
    # test
    batch(True)
    # batch_for_pbmap_test(4) # 2 4 17
