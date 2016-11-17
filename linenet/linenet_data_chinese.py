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

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import threshold

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_tar', 'data/chinese1.tar.gz',
                           """Path to the Sketch data file.""")
tf.app.flags.DEFINE_string('data_dir', 'data/chinese1',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_integer('image_width', 96, # 48-24-12-6
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 96, # 48-24-12-6
                            """Image Height.""")
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")


class BatchManager(object):
    def __init__(self, num_max=-1):
        # untar sketch file
        with tarfile.open(FLAGS.data_tar, 'r:gz') as tar:
            tar.extractall(FLAGS.data_dir)

        # read all svg files
        self._svg_list = []
        for root, _, files in os.walk(FLAGS.data_dir):
            for file in files:
                if not file.lower().endswith('svg_pre'):
                    continue

                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    svg = f.read().format(w=FLAGS.image_width, h=FLAGS.image_height)
                    svg_xml = et.fromstring(svg)
                    self._svg_list.append(svg_xml)
        
        # delete data
        tf.gfile.DeleteRecursively(FLAGS.data_dir)
        self._next_svg_id = 0
        
        self.num_examples_per_epoch = len(self._svg_list)
        self.num_epoch = 1
        
        d = 2 if FLAGS.use_two_channels else 1
        self.s_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
        self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, d], dtype=np.float)
        self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1], dtype=np.float)
    

    def _next_svg(self):
        svg_xml = copy.deepcopy(self._svg_list[self._next_svg_id])
        svg = et.tostring(svg_xml, method='xml')
        s_png = cairosvg.svg2png(bytestring=svg)
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) / 255.0
        
        # # debug
        # plt.imshow(s, cmap=plt.cm.gray)
        # plt.show()

        # leave only one path
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
        point_id = np.random.randint(len(line_ids[0]))
        px, py = line_ids[0][point_id], line_ids[1][point_id]
            
        # for next access
        self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
        if self._next_svg_id == 0:
            self.num_epoch = self.num_epoch + 1
            shuffle(self._svg_list)
                        
        return s, y, px, py

    
    def batch(self):
        for i in xrange(FLAGS.batch_size):
            s, y, px, py = self._next_svg()

            self.s_batch[i,:,:,:] = np.reshape(s, [FLAGS.image_height, FLAGS.image_width, 1])
            self.y_batch[i,:,:,:] = np.reshape(y, [FLAGS.image_height, FLAGS.image_width, 1])
            
            if FLAGS.use_two_channels:
                self.x_batch[i,:,:,0] = s
                x_point = np.zeros(s.shape)
                x_point[px, py] = 1.0
                self.x_batch[i,:,:,1] = x_point
            else:
                x = s / FLAGS.intensity_ratio
                x[px, py] = 1.0
                self.x_batch[i,:,:,:] = np.reshape(x, [FLAGS.image_height, FLAGS.image_width, 1])

        return self.s_batch, self.x_batch, self.y_batch


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
