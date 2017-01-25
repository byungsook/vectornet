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
import io
from random import shuffle
import xml.etree.ElementTree as et
import copy
import multiprocessing.managers
import multiprocessing.pool
from functools import partial
import platform

import numpy as np
import matplotlib.pyplot as plt

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data/sketch',
                           """Path to the Sketch data directory.""")
tf.app.flags.DEFINE_integer('image_width', 128,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 96,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")

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

        if platform.system() == 'Windows':
            FLAGS.num_processors = 1 # doesn't support MP

        if FLAGS.num_processors > FLAGS.batch_size:
            FLAGS.num_processors = FLAGS.batch_size

        if FLAGS.num_processors == 1:
            self.s_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 2], dtype=np.float)
            self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
        else:
            class MPManager(multiprocessing.managers.SyncManager):
                pass
            MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)

            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)
            
            self.s_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 2], dtype=np.float)
            self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self._svg_batch = self._mpmanager.list(['' for _ in xrange(FLAGS.batch_size)])
            self._func = partial(train_set, svg_batch=self._svg_batch,
                                 s_batch=self.s_batch, x_batch=self.x_batch, y_batch=self.y_batch)

    def __del__(self):
        if FLAGS.num_processors > 1:
            self._pool.terminate() # or close
            self._pool.join()


    def batch(self):
        if FLAGS.num_processors == 1:
            svg_batch = []
            for i in xrange(FLAGS.batch_size):
                svg_batch.append(self._svg_list[self._next_svg_id])
                train_set(i, svg_batch, self.s_batch, self.x_batch, self.y_batch)
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    if FLAGS.min_prop > 0.001: FLAGS.min_prop = FLAGS.min_prop * 0.5
                    shuffle(self._svg_list)
        else:
            for i in xrange(FLAGS.batch_size):
                self._svg_batch[i] = self._svg_list[self._next_svg_id]
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    if FLAGS.min_prop > 0.001: FLAGS.min_prop = FLAGS.min_prop * 0.5
                    shuffle(self._svg_list)

            self._pool.map(self._func, range(FLAGS.batch_size))

        return self.s_batch, self.x_batch, self.y_batch


def train_set(i, svg_batch, s_batch, x_batch, y_batch):
    with open(svg_batch[i], 'r') as sf:
        svg = sf.read().format(w=FLAGS.image_width, h=FLAGS.image_height)

    s_png = cairosvg.svg2png(bytestring=svg)
    s_img = Image.open(io.BytesIO(s_png))
    s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(s)
    s = s / max_intensity

    # # debug
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.show()

    # leave only one path
    svg_xml = et.fromstring(svg)
    # the first child of [0] is title
    # num_paths = len(svg_xml[0]._children) - 1
    num_paths = len(svg_xml[0]) - 1
    
    path_id_list = np.random.permutation(xrange(1,num_paths+1))

    for path_id in path_id_list:
        # svg_xml[0]._children = [svg_xml[0]._children[path_id]]
        stroke = svg_xml[0][path_id]
        for c in reversed(xrange(1,num_paths+1)):
            if svg_xml[0][c] != stroke:
                svg_xml[0].remove(svg_xml[0][c])
        svg_new = et.tostring(svg_xml, method='xml')

        y_png = cairosvg.svg2png(bytestring=svg_new)
        y_img = Image.open(io.BytesIO(y_png))
        y = np.array(y_img)[:,:,3].astype(np.float) / max_intensity

        # # debug
        # plt.imshow(y, cmap=plt.cm.gray)
        # plt.show()

        # select arbitrary marking pixel
        line_ids = np.nonzero(y)
        num_line_pixels = len(line_ids[0])
        proportion = num_line_pixels / (FLAGS.image_width*FLAGS.image_height)

        # # debug
        # print(path_id, proportion)

        # check if valid stroke
        if num_line_pixels == 0 or proportion < FLAGS.min_prop:
            svg_xml = et.fromstring(svg)
            if num_line_pixels > 0:
                last_valid = y
        else:
            break

    if num_line_pixels == 0:
        y = last_valid
        line_ids = np.nonzero(y)
        num_line_pixels = len(line_ids[0])

    point_id = np.random.randint(num_line_pixels)
    px, py = line_ids[0][point_id], line_ids[1][point_id]

    s_batch[i,:,:,:] = np.reshape(s, [FLAGS.image_height, FLAGS.image_width, 1])
    y_batch[i,:,:,:] = np.reshape(y, [FLAGS.image_height, FLAGS.image_width, 1])

    x_batch[i,:,:,0] = s
    x_point = np.zeros(s.shape)
    x_point[px, py] = 1.0
    x_batch[i,:,:,1] = x_point


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('linenet'):
        working_path = os.path.join(current_path, 'vectornet/linenet')
        os.chdir(working_path)

    # parameters 
    tf.app.flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_processors = 1
    FLAGS.min_prop = 0.01

    batch_manager = BatchManager()
    s_batch, x_batch, y_batch = batch_manager.batch()
    
    for i in xrange(FLAGS.batch_size):
        plt.imshow(np.reshape(s_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        t = np.concatenate((x_batch, np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1])), axis=3)
        plt.imshow(t[i,:,:,:], cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()

    print('Done')
