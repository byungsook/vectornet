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
import urllib
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import threshold

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_url', 'https://www.dropbox.com/s/e5ugvxdci5kv9g2/sketches-06-04.zip?dl=1', #'https://www.dropbox.com/s/ujb7bnwf147zjbp/sketches-06-04-mini.zip?dl=1', # 
                           """Url to the Sketch data file.""")
tf.app.flags.DEFINE_string('data_zip', 'data/sketches-06-04.zip',
                           """Path to the Sketch data file.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Path to the Sketch data directory.""")
tf.app.flags.DEFINE_integer('image_size', 128, # 48-24-12-6
                            """Image Size.""")
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")
tf.app.flags.DEFINE_float('initial_min_ratio', 0.02,
                          """initial_min_ratio for minimum length of line""")
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")


SVG_TEMPLATE_START = """<svg width="{s}" height="{s}" viewBox="0 0 640 480" 
                        xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" 
                        xmlns:xlink="http://www.w3.org/1999/xlink"><g>"""
SVG_TEMPLATE_END = """</g></svg>"""


class BatchManager(object):
    def __init__(self):
        # download sketch file unless it exists
        if not os.path.isfile(FLAGS.data_zip):
            urllib.urlretrieve(FLAGS.data_url, FLAGS.data_zip)

            # unzip sketch file
            with zipfile.ZipFile(FLAGS.data_zip, 'r') as zip_ref:
                zip_ref.extractall(FLAGS.data_dir)

        # extract all svg list
        self._svg_list = []
        file_list_name = 'checked.txt'
        for root, _, files in os.walk(FLAGS.data_dir):
            if not file_list_name in files:
                continue
            file_list_path = os.path.join(root, file_list_name)
            with open(file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break
                    file_name = line.rstrip('\n') + '.svg'
                    file_path = os.path.join(root, file_name)
                    self._svg_list.append(file_path)
        
        shuffle(self._svg_list)
        self._next_svg_id = 0
        self._read_next = True
        self._path_list = []
        self._path_id = 0

        self.num_examples_per_epoch = len(self._svg_list)
        self.num_epoch = 1
        self.ratio = FLAGS.initial_min_ratio
        
        d = 2 if FLAGS.use_two_channels else 1
        self.s_batch = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
        self.x_batch = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, d], dtype=np.float)
        self.y_batch = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    
    
    def _next_svg(self):
        success = False
        while not success:
            if self._read_next:
                # open next svg
                file_path = self._svg_list[self._next_svg_id]
                
                # preprocess svg
                with open(file_path, 'r') as f:
                    # svg = f.read() or
                    
                    # scale image
                    svg = f.readline()
                    id_width = svg.find('width')
                    id_xmlns = svg.find('xmlns', id_width)
                    svg_size = 'width="{s}" height="{s}" viewBox="0 0 640 480" '.format(s=FLAGS.image_size)
                    svg = svg[:id_width] + svg_size + svg[id_xmlns:]
                    
                    # gather normal paths and remove thick white stroke
                    self._path_list = []
                    while True:
                        svg_line = f.readline()
                        if not svg_line: break

                        # remove thick white strokes
                        id_white_stroke = svg_line.find('#fff')
                        if id_white_stroke == -1:
                            # gather normal paths
                            if svg_line.find('path t=') >= 0:
                                self._path_list.append(svg_line)
                            svg = svg + svg_line
                            
                shuffle(self._path_list)
                self._path_id = 0
                
                # read preprocessed svg
                try:
                    x_png = cairosvg.svg2png(bytestring=svg)
                except Exception as e:
                    # print('error %s, file %s' % (e, file_path))
                    svg = svg + '</svg>'
                    x_png = cairosvg.svg2png(bytestring=svg)

                x_img = Image.open(io.BytesIO(x_png))
                self.x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
                self.x = threshold(self.x, threshmax=0.01, newval=1.0)

                # # debug
                # plt.imshow(self.x, cmap=plt.cm.gray)
                # plt.show()

                # for next svg
                self._read_next = False
                
            
            # check y
            path = self._path_list[self._path_id]
            y_svg = SVG_TEMPLATE_START.format(s=FLAGS.image_size) + path + SVG_TEMPLATE_END
            y_png = cairosvg.svg2png(bytestring=y_svg)
            y_img = Image.open(io.BytesIO(y_png))
            y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
            y = threshold(y, threshmax=0.01, newval=1.0)
            line_ids = np.nonzero(y)
            
            if len(line_ids[0]) / (FLAGS.image_size*FLAGS.image_size) < self.ratio:
                del self._path_list[self._path_id]
            else:
                self._path_id = self._path_id + 1
                success = True

            # if there is no remaining path
            #if success or self._path_id >= len(self._path_list):
            if self._path_id >= len(self._path_list):
                self._read_next = True
                self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
                if self._next_svg_id == 0:
                    self.num_epoch = self.num_epoch + 1
                    if self.ratio > 0.001:
                        self.ratio = self.ratio * 0.5

        return self.x, y, line_ids


    def batch(self):
        for i in xrange(FLAGS.batch_size):
            x, y, line_ids = self._next_svg()

            # s
            self.s_batch[i,:,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
            # y
            self.y_batch[i,:,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])
            # x
            point_id = np.random.randint(len(line_ids[0]))
            px, py = line_ids[0][point_id], line_ids[1][point_id]
            
            if FLAGS.use_two_channels:
                self.x_batch[i,:,:,0] = x
                x_point = np.zeros(x.shape)
                x_point[px, py] = 1.0
                self.x_batch[i,:,:,1] = x_point
            else:
                x = x / FLAGS.intensity_ratio
                x[px, py] = 1.0
                self.x_batch[i,:,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])

        return self.s_batch, self.x_batch, self.y_batch


def check_num_path(filepath):
    with open(filepath, 'r') as f:
        svg = f.read()
        num_paths = svg.count('path')
        print(filepath, num_paths)
    return num_paths


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('linenet'):
        working_path = os.path.join(current_path, 'vectornet/linenet')
        os.chdir(working_path)

    batch_manager = BatchManager()

    s_batch, x_batch, y_batch = batch_manager.batch()
    for i in xrange(FLAGS.batch_size):
        plt.imshow(np.reshape(s_batch[i,:], [FLAGS.image_size,FLAGS.image_size]), cmap=plt.cm.gray)
        plt.show()
        if FLAGS.use_two_channels:
            t = np.concatenate((x_batch, np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1])), axis=3)
            plt.imshow(t[i,:,:,:], cmap=plt.cm.gray)
        else:
            plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_size,FLAGS.image_size]), cmap=plt.cm.gray)
        plt.show()
        plt.imshow(np.reshape(y_batch[i,:], [FLAGS.image_size,FLAGS.image_size]), cmap=plt.cm.gray)
        plt.show()

    # for i in xrange(batch_manager.num_examples_per_epoch):
    #     batch_manager.batch()


    filelist = 'checked.txt'
    for root, _, files in os.walk(FLAGS.data_dir):
        if not filelist in files:
            continue
        filelistpath = os.path.join(root, filelist)
        avg_num_paths = 0
        num_files = 0
        with open(filelistpath, 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                filename = line.rstrip('\n') + '.svg'
                filepath = os.path.join(root, filename)
                avg_num_paths = avg_num_paths + check_num_path(filepath)
                num_files = num_files + 1
        avg_num_paths = avg_num_paths / num_files
        print('# files: %d, avg of # paths: %d' % (num_files, avg_num_paths))

    print('Done')