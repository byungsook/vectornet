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

import numpy as np
import matplotlib.pyplot as plt

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data/chinese1',
                           """Path to the chinese data directory.""")
tf.app.flags.DEFINE_integer('image_width', 128,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 128,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")
tf.app.flags.DEFINE_boolean('chinese1', True,
                            """whether chinese1 or not""")


class BatchManager(object):
    def __init__(self):
        # read all svg files
        self._next_svg_id = 0
        self._svg_list = []
        if FLAGS.file_list:
            file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
            with open(file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break

                    file_path = os.path.join(FLAGS.data_dir, line.rstrip())
                    self._svg_list.append(file_path)
                    # with open(file_path, 'r') as sf:
                    #     svg = sf.read()
                    #     self._svg_list.append(svg)

        else:
            for root, _, files in os.walk(FLAGS.data_dir):
                for file in files:
                    if not file.lower().endswith('svg_pre'):
                        continue

                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        svg = f.read()
                        self._svg_list.append(svg)

        self.num_examples_per_epoch = len(self._svg_list)
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
            self._svg_batch = self._mpmanager.list(['' for _ in xrange(FLAGS.batch_size)])
            self._func = partial(train_set, svg_batch=self._svg_batch, x_batch=self.x_batch, y_batch=self.y_batch)


    def __del__(self):
        if FLAGS.num_processors > 1:
            self._pool.terminate() # or close
            self._pool.join()


    def batch(self):
        if FLAGS.num_processors == 1:
            svg_batch = []
            for i in xrange(FLAGS.batch_size):
                svg_batch.append(self._svg_list[self._next_svg_id])
                train_set(i, svg_batch, self.x_batch, self.y_batch)
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    shuffle(self._svg_list)
        else:
            for i in xrange(FLAGS.batch_size):
                self._svg_batch[i] = self._svg_list[self._next_svg_id]
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    shuffle(self._svg_list)

            self._pool.map(self._func, range(FLAGS.batch_size))

        return self.x_batch, self.y_batch


def train_set(batch_id, svg_batch, x_batch, y_batch):
    while True:
        if FLAGS.chinese1:
            if FLAGS.transform:
                r = np.random.randint(-45, 45)
                # s_sign = np.random.choice([1, -1], 1)[0]
                s_sign = -1
                s = 1.75 * np.random.random_sample(2) + 0.5 # [0.5, 2)
                s[1] = s[1] * s_sign
                t = np.random.randint(-20, 20, 2)
                if s_sign == 1:
                    t[1] = t[1] + 124
                else:
                    t[1] = t[1] - 900
            else:
                r = 0
                s = [1, -1]
                t = [0, -900]
        else:
            if FLAGS.transform:
                r = np.random.randint(-45, 45)
                # s_sign = np.random.choice([1, -1], 1)[0]
                s_sign = 1
                s = 1.75 * np.random.random_sample(2) + 0.25 # [0.25, 2)
                s[1] = s[1] * s_sign
                t = np.random.randint(-10, 10, 2)
                if s_sign == -1:
                    t[1] = t[1] - 109
            else:
                r = 0
                s = [1, 1]
                t = [0, 0]
        
        with open(svg_batch[batch_id], 'r') as sf:
            svg = sf.read().format(
                w=FLAGS.image_width, h=FLAGS.image_height,
                r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])

        x_png = cairosvg.svg2png(bytestring=svg)
        x_img = Image.open(io.BytesIO(x_png))
        x = np.array(x_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(x)
        
        if max_intensity == 0:
            continue
        else:
            x = x / max_intensity

        y = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.bool)
        stroke_list = []
        if FLAGS.chinese1:
            svg_xml = ET.fromstring(svg)
            num_paths = len(svg_xml[0]._children)

            for i in xrange(num_paths):
                svg_xml = ET.fromstring(svg)
                svg_xml[0]._children = [svg_xml[0]._children[i]]
                svg_one_stroke = ET.tostring(svg_xml, method='xml')

                stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
                stroke_img = Image.open(io.BytesIO(stroke_png))
                stroke = (np.array(stroke_img)[:,:,3] > 0)

                # # debug
                # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
                # plt.imshow(stroke_img, cmap=plt.cm.gray)
                # plt.show()

                stroke_list.append(stroke)
        else:
            pass

        for i in xrange(num_paths-1):
            for j in xrange(i+1, num_paths):
                intersect = np.logical_and(stroke_list[i], stroke_list[j])
                y = np.logical_or(intersect, y)

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

        y = np.multiply(x, y)

        # # debug
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(x, cmap=plt.cm.gray)
        # plt.subplot(122)
        # plt.imshow(y, cmap=plt.cm.gray)
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        # plt.show()

        break

    x_batch[batch_id,:,:,0] = x
    y_batch[batch_id,:,:,0] = y


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('linenet'):
        working_path = os.path.join(current_path, 'vectornet/linenet')
        os.chdir(working_path)

    # parameters 
    tf.app.flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_processors = 1
    FLAGS.transform = True

    batch_manager = BatchManager()
    x_batch, y_batch = batch_manager.batch()
    
    for i in xrange(FLAGS.batch_size):
        plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()

    print('Done')
