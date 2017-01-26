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
import scipy.ndimage

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../data/hand',
                           """Path to the chinese data directory.""")
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
            # file_int_list_path = os.path.join(FLAGS.data_dir, 'inter.txt')
            # fff = open(file_int_list_path, 'w')
            with open(file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break

                    file_path = os.path.join(FLAGS.data_dir, line.rstrip())
                    self._svg_list.append(file_path)
            #         if find_intersection(file_path):
            #             self._svg_list.append(file_path)
            #             fff.write(line)
            # fff.close()
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


def find_intersection(svg_file_path):
    with open(svg_file_path, 'r') as sf:
        svg = sf.read()

    c_start = svg.find('<!--') + 4
    c_end = svg.find('-->', c_start)
    w, h = svg[c_start:c_end].split()
    w = int(round(float(w)))
    h = int(round(float(h)))
    w_end = c_end
    
    num_lines = svg.count('\n')
    num_strokes = int((num_lines - 5) / 2) # polyline start 6
    # stroke_width = np.random.randint(FLAGS.max_stroke_width) + 1
    stroke_width = 1

    svg = svg.format(
        w=w, h=h,
        bx=0, by=0, bw=w, bh=h,
        sw=stroke_width)

    y = np.zeros([h, w], dtype=np.bool)
    stroke_list = []
    for stroke_id in xrange(num_strokes):

        # leave only one path
        svg_xml = ET.fromstring(svg)
        if sys.version_info > (2,7):
            stroke = svg_xml[0][stroke_id]
            for c in reversed(xrange(num_strokes)):
                if svg_xml[0][c] != stroke:
                    svg_xml[0].remove(svg_xml[0][c])
        else:
            svg_xml[0]._children = [svg_xml[0]._children[stroke_id]]
        svg_one_stroke = ET.tostring(svg_xml, method='xml')

        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke = np.array(stroke_img)[:,:,3].astype(np.float)

        # # debug
        # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
        # plt.imshow(stroke_img, cmap=plt.cm.gray)
        # plt.show()

        stroke_list.append(stroke)

    for i in xrange(num_strokes-1):
        for j in xrange(i+1, num_strokes):
            intersect = np.logical_and(stroke_list[i], stroke_list[j])
            y = np.logical_or(intersect, y)
            
            # # debug
            # if np.sum(intersect) > 0:
            #     plt.figure()
            #     plt.subplot(221)
            #     plt.imshow(stroke_list[i], cmap=plt.cm.gray)
            #     plt.subplot(222)
            #     plt.imshow(stroke_list[j], cmap=plt.cm.gray)
            #     plt.subplot(223)
            #     plt.imshow(intersect, cmap=plt.cm.gray)
            #     plt.subplot(224)
            #     plt.imshow(y, cmap=plt.cm.gray)
            #     mng = plt.get_current_fig_manager()
            #     mng.full_screen_toggle()
            #     plt.show()

    num_intersections = np.sum(y)
    return num_intersections > 0


def train_set(batch_id, svg_batch, x_batch, y_batch):
    np.random.seed()
    with open(svg_batch[batch_id], 'r') as sf:
        svg = sf.read()

    c_start = svg.find('<!--') + 4
    c_end = svg.find('-->', c_start)
    w, h = svg[c_start:c_end].split()
    w = float(w)
    h = float(h)
    w_end = c_end

    if w < FLAGS.image_width:
        w = FLAGS.image_width
    
    num_lines = svg.count('\n')
    num_strokes = int((num_lines - 5) / 2) # polyline start 6
    stroke_width = 3

    g_start = svg.find('fill="none"')
    svg = svg[:g_start] + 'transform="rotate({r},512,512) scale({sx},{sy}) translate({tx},{ty})" ' + svg[g_start:]
    if FLAGS.transform:
        r = np.random.randint(-10, 10)
        by = np.random.rand()*20.0 - 10.0
    else:
        r = 0
        by = 0
    
    svg_all = svg.format(
        w=w, h=h,
        bx=0, by=by, bw=w, bh=h,
        sw=stroke_width,
        r=r, sx=1, sy=1, tx=0, ty=0)

    stroke_list = []
    # stroke_bbox_list = []
    stroke_id_list = np.random.permutation(xrange(0,num_strokes))
    for stroke_id in stroke_id_list:
        # c_end = w_end

        # for _ in xrange(stroke_id+1):
        #     c_start = svg.find('<!--', c_end) + 4
        #     c_end = svg.find('-->', c_start)
        # x1, x2, _, _ = svg[c_start:c_end].split()

        # min_bx = max(0, float(x2)-FLAGS.image_width)
        # max_bx = min(w-FLAGS.image_width, float(x1))
        # stroke_bbox_list.append((min_bx,max_bx))

        # leave only one path
        svg_xml = ET.fromstring(svg_all)
        if sys.version_info > (2,7):
            stroke = svg_xml[0][stroke_id]
            for c in reversed(xrange(num_strokes)):
                if svg_xml[0][c] != stroke:
                    svg_xml[0].remove(svg_xml[0][c])
        else:
            svg_xml[0]._children = [svg_xml[0]._children[stroke_id]]
        svg_one_stroke = ET.tostring(svg_xml, method='xml')

        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke = np.array(stroke_img)[:,:,3].astype(np.float)

        # # debug
        # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
        # plt.imshow(stroke_img, cmap=plt.cm.gray)
        # plt.show()

        stroke_list.append(stroke)

    
    y = np.zeros([stroke.shape[0], stroke.shape[1]], dtype=np.bool)
    bx = int(np.random.rand() * (w-FLAGS.image_width))
    find_intersection = False
    for i in xrange(num_strokes-1):
        for j in xrange(i+1, num_strokes):
            intersect = np.logical_and(stroke_list[i], stroke_list[j])
            y = np.logical_or(intersect, y)
            num_intersect = np.sum(intersect)

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

            if num_intersect > 0 and not find_intersection:
                cols = np.any(intersect, axis=0)
                cmin, cmax = np.where(cols)[0][[0, -1]]
                cmin = max(0, cmin-5)
                cmax = min(w, cmax+5)
                min_bx = max(0, cmax-FLAGS.image_width)
                max_bx = min(w-FLAGS.image_width, cmin)
                bx = int(np.random.rand() * (max_bx - min_bx) + min_bx)
                find_intersection = True

    y_crop = y[:,bx:bx+FLAGS.image_width]

    svg_crop = svg.format(
        w=FLAGS.image_width, h=FLAGS.image_height,
        bx=bx, by=by, bw=FLAGS.image_width, bh=FLAGS.image_height,
        sw=stroke_width,
        r=r, sx=1, sy=1, tx=0, ty=0)

    s_png = cairosvg.svg2png(bytestring=svg_crop.encode('utf-8'))
    s_img = Image.open(io.BytesIO(s_png))

    x = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(x)
    if max_intensity > 0:
        x /= max_intensity
    
    # y = np.multiply(x, y_crop) * 1000
    y = y_crop.astype(np.float) * 1000

    # debug
    plt.figure()
    plt.subplot(131)
    plt.imshow(s_img)
    plt.subplot(132)
    plt.imshow(x, cmap=plt.cm.gray)
    plt.subplot(133)
    plt.imshow(y, cmap=plt.cm.gray)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

    x_batch[batch_id,:,:,0] = x
    y_batch[batch_id,:,:,0] = y


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('ovnet'):
        working_path = os.path.join(current_path, 'vectornet/ovnet')
        os.chdir(working_path)

    # parameters 
    tf.app.flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_processors = 1
    FLAGS.transform = True

    batch_manager = BatchManager()
    while True:
        x_batch, y_batch = batch_manager.batch()
    
    for i in xrange(FLAGS.batch_size):
        plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()

    print('Done')
