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
import tarfile
import xml.etree.ElementTree as et
import copy
import multiprocessing.managers
import multiprocessing.pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import threshold

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_tar', 'data/chinese1.tar.gz',
                           """Path to the Sketch data file.""")
tf.app.flags.DEFINE_string('data_dir', 'data/chinese1',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_integer('image_width', 128, # 48-24-12-6
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 128, # 48-24-12-6
                            """Image Height.""")
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")
tf.app.flags.DEFINE_integer('num_processors', 8,
                            """# of processors for batch generation.""")


class BatchManager(object):
    def __init__(self):
        # # untar sketch file
        # with tarfile.open(FLAGS.data_tar, 'r:gz') as tar:
        #     tar.extractall(FLAGS.data_dir)

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
                    with open(file_path, 'r') as sf:
                        svg = sf.read()
                        self._svg_list.append(svg)

        else:
            for root, _, files in os.walk(FLAGS.data_dir):
                for file in files:
                    if not file.lower().endswith('svg_pre'):
                        continue

                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        svg = f.read()
                        self._svg_list.append(svg)

        # # delete data
        # tf.gfile.DeleteRecursively(FLAGS.data_dir)

        self.num_examples_per_epoch = len(self._svg_list)
        self.num_epoch = 1

        d = 2 if FLAGS.use_two_channels else 1

        if FLAGS.num_processors > FLAGS.batch_size:
            FLAGS.num_processors = FLAGS.batch_size
            
        if FLAGS.num_processors == 1:
            self.s_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, d], dtype=np.float)
            self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
        else:
            class MPManager(multiprocessing.managers.SyncManager):
                pass
            MPManager.register('np_empty', np.empty, multiprocessing.managers.ArrayProxy)

            self._mpmanager = MPManager()
            self._mpmanager.start()
            self._pool = multiprocessing.pool.Pool(processes=FLAGS.num_processors)
            
            self.s_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
            self.x_batch = self._mpmanager.np_empty([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, d], dtype=np.float)
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
            for i in xrange(FLAGS.batch_size):
                train_set(0, [self._svg_list[self._next_svg_id]], self.s_batch, self.x_batch, self.y_batch)
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
    while True:
        if FLAGS.transform:
            r = np.random.randint(-180, 181)
            s_sign = np.random.choice([1, -1], 1)[0]
            s = 1.75 * np.random.random_sample(2) + 0.25 # [0.25, 2)
            s[1] = s[1] * s_sign
            t = np.random.randint(-100, 100, 2) # [-100, 100]
            if s_sign == 1:
                t[1] = t[1] + 124
            else:
                t[1] = t[1] - 900
        else:
            r = 0
            s = [1, -1]
            t = [0, -900]
        
        svg = svg_batch[i].format(
                w=FLAGS.image_width, h=FLAGS.image_height,
                r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        s_png = cairosvg.svg2png(bytestring=svg)
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) / 255.0
        
        if len(np.nonzero(s)[0]) == 0:
            continue
    
        # # debug
        # plt.imshow(s, cmap=plt.cm.gray)
        # plt.show()

        # leave only one path
        svg_xml = et.fromstring(svg)
        path_id = np.random.randint(len(svg_xml[0]._children))
        svg_xml[0]._children = [svg_xml[0]._children[path_id]]
        
        svg = et.tostring(svg_xml, method='xml')
        y_png = cairosvg.svg2png(bytestring=svg)
        y_img = Image.open(io.BytesIO(y_png))
        y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
        
        # # debug
        # plt.imshow(y, cmap=plt.cm.gray)
        # plt.show()

        # select arbitrary marking pixel
        line_ids = np.nonzero(y)
        if len(line_ids[0]) == 0:
            continue
        else:
            break

    point_id = np.random.randint(len(line_ids[0]))
    px, py = line_ids[0][point_id], line_ids[1][point_id]
    
    s_batch[i,:,:,:] = np.reshape(s, [FLAGS.image_height, FLAGS.image_width, 1])
    y_batch[i,:,:,:] = np.reshape(y, [FLAGS.image_height, FLAGS.image_width, 1])
    
    if FLAGS.use_two_channels:
        x_batch[i,:,:,0] = s
        x_point = np.zeros(s.shape)
        x_point[px, py] = 1.0
        x_batch[i,:,:,1] = x_point
    else:
        x = s / FLAGS.intensity_ratio
        x[px, py] = 1.0
        x_batch[i,:,:,:] = np.reshape(x, [FLAGS.image_height, FLAGS.image_width, 1])


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('linenet'):
        working_path = os.path.join(current_path, 'vectornet/linenet')
        os.chdir(working_path)

    batch_manager = BatchManager()
    s_batch, x_batch, y_batch = batch_manager.batch()
    
    for i in xrange(FLAGS.batch_size):
        plt.imshow(np.reshape(s_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        if FLAGS.use_two_channels:
            t = np.concatenate((x_batch, np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1])), axis=3)
            plt.imshow(t[i,:,:,:], cmap=plt.cm.gray)
        else:
            plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width]), cmap=plt.cm.gray)
        plt.show()
        
    print('Done')
