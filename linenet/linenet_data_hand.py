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
import xml.etree.ElementTree as et
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
tf.app.flags.DEFINE_string('data_dir', 'data/hand',
                           """Path to the hand writing data directory.""")
tf.app.flags.DEFINE_integer('image_size', 128,
                            """Image size.""")
tf.app.flags.DEFINE_integer('image_width', 128,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 128,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")


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
            self.s_batch = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
            self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 2], dtype=np.float)
            self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
        else:
            class MPManager(multiprocessing.managers.SyncManager):
                pass
            MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)

            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)
            
            self.s_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 2], dtype=np.float)
            self.y_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
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
                    shuffle(self._svg_list)
        else:
            for i in xrange(FLAGS.batch_size):
                self._svg_batch[i] = self._svg_list[self._next_svg_id]
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    shuffle(self._svg_list)

            self._pool.map(self._func, range(FLAGS.batch_size))

        return self.s_batch, self.x_batch, self.y_batch


def train_set(i, svg_batch, s_batch, x_batch, y_batch):
    with open(svg_batch[i], 'r') as sf:
        svg = sf.read()

    c_start = svg.find('<!--') + 4
    c_end = svg.find('-->', c_start)
    w = float(svg[c_start:c_end].split()[0])
    w_end = c_end
    
    num_lines = svg.count('\n')
    num_strokes = int((num_lines - 5) / 2) # polyline start 6

    # stroke_id = np.random.randint(num_strokes)
    stroke_id_list = np.random.permutation(xrange(0,num_strokes))
    for stroke_id in stroke_id_list:
        c_end = w_end

        for _ in xrange(stroke_id+1):
            c_start = svg.find('<!--', c_end) + 4
            c_end = svg.find('-->', c_start)
        x1, x2, _, _ = svg[c_start:c_end].split()

        min_bx = max(0, float(x2)-FLAGS.image_size)
        max_bx = min(w-FLAGS.image_size, float(x1))
        bx = np.random.rand() * (max_bx - min_bx) + min_bx

        svg_crop = svg.format(
            w=FLAGS.image_size, h=FLAGS.image_size,
            bx=bx, by=0, bw=FLAGS.image_size, bh=FLAGS.image_size)
        s_png = cairosvg.svg2png(bytestring=svg_crop.encode('utf-8'))
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)

        if max_intensity == 0:
            continue
        else:
            s = s / max_intensity

        # leave only one path
        svg_xml = et.fromstring(svg_crop)
        if sys.version_info > (2,7):
            stroke = svg_xml[0][stroke_id]
            for c in reversed(xrange(num_strokes)):
                if svg_xml[0][c] != stroke:
                    svg_xml[0].remove(svg_xml[0][c])
        else:
            svg_xml[0]._children = [svg_xml[0]._children[stroke_id]]
        svg_new = et.tostring(svg_xml, method='xml')

        y_png = cairosvg.svg2png(bytestring=svg_new)
        y_img = Image.open(io.BytesIO(y_png))
        y = np.array(y_img)[:,:,3].astype(np.float) / max_intensity
        
        # # debug
        # svg_all = svg.format(
        #     w=w, h=FLAGS.image_size,
        #     bx=0, by=0, bw=w, bh=FLAGS.image_size)
        # s_all_png = cairosvg.svg2png(bytestring=svg_all.encode('utf-8'))
        # s_all_img = Image.open(io.BytesIO(s_all_png))
        # print('stroke_id', stroke_id)
        # print('min_bx, max_bx, bx', min_bx, max_bx, bx)
        # plt.figure()
        # plt.subplot2grid((3,2), (0,0), colspan=2)
        # plt.imshow(s_all_img)
        # plt.subplot2grid((3,2), (1,0))
        # plt.imshow(s_img)
        # plt.subplot2grid((3,2), (1,1))
        # plt.imshow(s, cmap=plt.cm.gray)
        # plt.subplot2grid((3,2), (2,0))
        # plt.imshow(y_img)
        # plt.subplot2grid((3,2), (2,1))
        # plt.imshow(y, cmap=plt.cm.gray)
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        # plt.show()

        # select arbitrary marking pixel
        line_ids = np.nonzero(y)
        num_line_pixels = len(line_ids[0])

        proportion = num_line_pixels / (FLAGS.image_width*FLAGS.image_height)

        # # debug
        # print(path_id, proportion)

        # check if valid stroke
        if num_line_pixels == 0 or proportion < FLAGS.min_prop:
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

    s_batch[i,:,:,:] = np.reshape(s, [FLAGS.image_size, FLAGS.image_size, 1])
    y_batch[i,:,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])

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
        plt.imshow(np.reshape(s_batch[i,:], [FLAGS.image_size, FLAGS.image_size]), cmap=plt.cm.gray)
        plt.show()
        t = np.concatenate((x_batch, np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1])), axis=3)
        plt.imshow(t[i,:,:,:], cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_size, FLAGS.image_size]), cmap=plt.cm.gray)
        plt.show()

    print('Done')
