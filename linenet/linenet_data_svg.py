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

import numpy as np
import matplotlib.pyplot as plt

import cairosvg
from PIL import Image

import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data/sketches',
                           """Path to the Sketch data directory.""")
tf.app.flags.DEFINE_integer('image_size', 96, # 48-24-12-6
                            """Image Size.""")
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")


SVG_TEMPLATE_START = """<svg width="{s}" height="{s}" viewBox="0 0 640 480" 
                        xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" 
                        xmlns:xlink="http://www.w3.org/1999/xlink"><g>"""
SVG_TEMPLATE_END = """</g></svg>"""


class BatchManager(object):
    def __init__(self):        
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
        self._valid_path_dict = {}

        self.num_examples_per_epoch = len(self._svg_list)
        self.num_epoch = 1        
        
        batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1]
        self.s_batch = np.zeros(batch_shape, dtype=np.float)
        self.x_batch = np.zeros(batch_shape, dtype=np.float)
        self.y_batch = np.zeros(batch_shape, dtype=np.float)
    
    
    def batch(self):
        debug = False
        
        # preprocessing
        while True:
            file_path = self._svg_list[self._next_svg_id]
            with open(file_path, 'r') as f:
                # svg = f.read() or
                
                # scale image
                svg = f.readline()
                id_width = svg.find('width')
                id_xmlns = svg.find('xmlns', id_width)
                svg_size = 'width="{s}" height="{s}" viewBox="0 0 640 480" '.format(s=FLAGS.image_size)
                svg = svg[:id_width] + svg_size + svg[id_xmlns:]
                
                # gather normal paths and remove thick white stroke
                path_list = []
                while True:
                    svg_line = f.readline()
                    if not svg_line: break

                    # remove thick white strokes
                    id_white_stroke = svg_line.find('#fff')
                    if id_white_stroke == -1:
                        # gather normal paths
                        if svg_line.find('path t=') >= 0:
                            path_list.append(svg_line)
                        svg = svg + svg_line

            self._next_svg_id = (self._next_svg_id + 1) % len(self._svg_list)
            if self._next_svg_id == 0:
                self.num_epoch = self.num_epoch + 1

            try:
                x_png = cairosvg.svg2png(bytestring=svg)
            except Exception as e:
                print('error %e, file %s' % (e, file_path))
                pass
            else:
                break


        x_img = Image.open(io.BytesIO(x_png))
        x = np.array(x_img)[:,:,3].astype(np.float) / 255.0
        
        # debug
        if debug:
            plt.imshow(x, cmap=plt.cm.gray)
            plt.show()
        
        s = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
        self.s_batch = np.tile(s, (FLAGS.batch_size, 1, 1, 1))

        x = x / FLAGS.intensity_ratio

        i = 0
        use_dict = False
        if file_path in self._valid_path_dict:
            path_list = self._valid_path_dict[file_path]
            use_dict = True
        while True:
            # TODO: new seed for MP
            path_id = np.random.randint(len(path_list))
            path = path_list[path_id]
            svg_one_line = SVG_TEMPLATE_START.format(s=FLAGS.image_size) + path + SVG_TEMPLATE_END

            y_png = cairosvg.svg2png(bytestring=svg_one_line)
            y_img = Image.open(io.BytesIO(y_png))
            y = np.array(y_img)[:,:,3].astype(np.float) / 255.0
            self.y_batch[i,:,:,:] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])

            line_ids = np.nonzero(y)
            
            # debug
            if debug: print('path', path_id, 'nonzero', len(line_ids[0]))
            
            if not use_dict and len(line_ids[0]) / (FLAGS.image_size*FLAGS.image_size) < 0.002:
                del path_list[path_id]
                continue

            # debug
            if debug:
                plt.imshow(y, cmap=plt.cm.gray)
                plt.show()

            point_id = np.random.randint(len(line_ids[0]))
            px, py = line_ids[0][point_id], line_ids[1][point_id]
            
            tmp = x[px, py]
            x[px, py] = 1.0
            self.x_batch[i,:,:,:] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
            x[px, py] = tmp

            # debug
            if debug:
                plt.imshow(np.reshape(self.x_batch[i,:,:,:], [FLAGS.image_size, FLAGS.image_size]), cmap=plt.cm.gray)
                plt.show()

            i = i + 1
            
            # debug
            if debug: print('batch', i)

            if i == FLAGS.batch_size:
                if not use_dict:
                    self._valid_path_dict[file_path] = path_list
                break

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
        plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_size,FLAGS.image_size]), cmap=plt.cm.gray)
        plt.show()

    # filelist = 'checked.txt'
    # for root, _, files in os.walk(FLAGS.data_dir):
    #     if not filelist in files:
    #         continue
    #     filelistpath = os.path.join(root, filelist)
    #     avg_num_paths = 0
    #     num_files = 0
    #     with open(filelistpath, 'r') as f:
    #         while True:
    #             line = f.readline()
    #             if not line: break
    #             filename = line.rstrip('\n') + '.svg'
    #             filepath = os.path.join(root, filename)
    #             avg_num_paths = avg_num_paths + check_num_path(filepath)
    #             num_files = num_files + 1
    #             s_batch, x_batch, y_batch = batch(filepath)
    #             for i in xrange(FLAGS.batch_size):
    #                 plt.imshow(np.reshape(x_batch[i,:], [FLAGS.image_size,FLAGS.image_size]), cmap=plt.cm.gray)
    #                 plt.show()
    #     avg_num_paths = avg_num_paths / num_files
    #     print('# files: %d, avg of # paths: %d' % (num_files, avg_num_paths))

    # print('Done')