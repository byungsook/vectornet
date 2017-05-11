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
import random
import xml.etree.ElementTree as et
import threading
import multiprocessing
import signal
import sys
from datetime import datetime
import time

import numpy as np
import matplotlib.pyplot as plt

import cairosvg
from PIL import Image
from skimage import transform

import tensorflow as tf


# parameters
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 8,
                     """Number of images to process in a batch.""")
flags.DEFINE_string('data_dir', '../data/fidelity',
                    """Path to the Sketch data directory.""")
flags.DEFINE_integer('image_width', 128,
                     """Image Width.""")
flags.DEFINE_integer('image_height', 128,
                     """Image Height.""")
flags.DEFINE_integer('num_threads', multiprocessing.cpu_count(),
                     """# of threads for batch generation.""")
flags.DEFINE_boolean('use_two_channels', True,
                     """use two channels for input""")
FLAGS = flags.FLAGS


class BatchManager(object):
    def __init__(self):
        # read all svg files
        self._next_svg_id = 0
        self._data_list = []
        if FLAGS.file_list:
            file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
            with open(file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break

                    file_path = os.path.join(FLAGS.data_dir, line.rstrip())
                    self._data_list.append(file_path)
                    # with open(file_path, 'r') as sf:
                    #     svg = sf.read()
                    #     self._data_list.append(svg)

        else:
            for root, _, files in os.walk(FLAGS.data_dir):
                for file in files:
                    if not file.lower().endswith('svg_pre'):
                        continue

                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        svg = f.read()
                        self._data_list.append(svg)

        self.num_examples_per_epoch = len(self._data_list)
        self.num_epoch = 1

        FLAGS.num_threads = np.amin([FLAGS.num_threads, FLAGS.batch_size])


        image_shape = [FLAGS.image_height, FLAGS.image_width, 1]
        input_shape = [FLAGS.image_height, FLAGS.image_width, 2]

        self._q = tf.FIFOQueue(FLAGS.batch_size*10, [tf.float32, tf.float32], 
                               shapes=[input_shape, image_shape])

        self._x = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self._y = tf.placeholder(dtype=tf.float32, shape=image_shape)
        self._enqueue = self._q.enqueue([self._x, self._y])

    def __del__(self):
        try:
            self.stop_thread()
        except AttributeError:
            pass

    def batch(self):
        return self._q.dequeue_many(FLAGS.batch_size)

    def start_thread(self, sess):
        # Main thread: create a coordinator.
        self._coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, x, y, data_list, FLAGS):
            while not coord.should_stop():
                file_path = random.choice(data_list)
                x_, y_ = preprocess(file_path, FLAGS)
                sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self._threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(sess, 
                                                self._enqueue,
                                                self._coord,
                                                self._x,
                                                self._y,
                                                self._data_list,
                                                FLAGS)
                                          ) for i in xrange(FLAGS.num_threads)]

        # define signal handler
        def signal_handler(signum,frame):
            #print "stop training, save checkpoint..."
            #saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            sess.run(self._q.close(cancel_pending_enqueues=True))
            self._coord.request_stop()
            self._coord.join(self._threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self._threads:
            t.start()

    def stop_thread(self):
        self._coord.request_stop()
        self._coord.join(self._threads)


def preprocess(file_path, FLAGS):
    with open(file_path, 'r') as sf:
        svg = sf.read().format(w=1024, h=1024)

    s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    s_img = Image.open(io.BytesIO(s_png))
    s = np.array(s_img)[:,:,3].astype(np.float)

    # # debug
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.show()

    num_paths = svg.count('<path')
    path_list = []
    end = 0
    for i in xrange(num_paths):
        start = svg.find('<path', end)
        end = svg.find('/>', start) + 2
        path_list.append([start,end])
    
    path_id = np.random.randint(num_paths)

    y_svg = svg[:path_list[0][0]] + svg[path_list[path_id][0]:path_list[path_id][1]] + svg[path_list[-1][1]:]
    y_png = cairosvg.svg2png(bytestring=y_svg.encode('utf-8'))
    y_img = Image.open(io.BytesIO(y_png))
    y = np.array(y_img)[:,:,3].astype(np.float)

    # # debug
    # plt.imshow(y, cmap=plt.cm.gray)
    # plt.show()

    # random flip and rotate
    flip = (np.random.rand() > 0.5)
    if flip:
        y_rotate = np.fliplr(y)
        s_rotate = np.fliplr(s)
    else:
        y_rotate = y
        s_rotate = s
    r = np.random.rand() * 360.0
    y_rotate = transform.rotate(y_rotate, r, order=3, mode='symmetric')
    y = y_rotate
        
    s_rotate = transform.rotate(s_rotate, r, order=3, mode='symmetric')
    s = s_rotate

    # bbox
    y_nz = np.nonzero(y)
    y_h = [np.amin(y_nz[0]), np.amax(y_nz[0])+1]
    y_w = [np.amin(y_nz[1]), np.amax(y_nz[1]+1)]

    # # debug
    # plt.imshow(y[y_h[0]:y_h[1], y_w[0]:y_w[1]], cmap=plt.cm.gray)
    # plt.show()

    while True:
        try:
            if y_h[1] - y_h[0] < FLAGS.image_height:
                h = np.random.randint(low=max(0, y_h[1]-FLAGS.image_height), high=y_h[0]+1)
            else:
                h = np.random.randint(low=y_h[0], high=y_h[1]-FLAGS.image_height)
            if y_w[1] - y_w[0] < FLAGS.image_width:
                w = np.random.randint(low=max(0, y_w[1]-FLAGS.image_height), high=y_w[0]+1)
            else:
                w = np.random.randint(low=y_w[0], high=y_w[1]-FLAGS.image_height)
        except:
            continue

        y_crop = y[h:h+FLAGS.image_height, w:w+FLAGS.image_width]

        hs = int(FLAGS.image_height*0.1)
        ws = int(FLAGS.image_width*0.1)
        he = hs + int(FLAGS.image_height*0.8)
        we = ws + int(FLAGS.image_width*0.8)

        line_ids = np.nonzero(y_crop[hs:he,ws:we])
        num_line_pixels = len(line_ids[0])

        if num_line_pixels > 0:
            break

    s = s[h:h+FLAGS.image_height, w:w+FLAGS.image_width]
    max_intensity = np.amax(s)
    s = s / max_intensity

    y = y_crop / max_intensity

    line_ids = np.nonzero(y)
    num_line_pixels = len(line_ids[0])
    point_id = np.random.randint(num_line_pixels)
    ph, pw = line_ids[0][point_id], line_ids[1][point_id]

    y = np.reshape(y, [FLAGS.image_height, FLAGS.image_width, 1])
    x = np.zeros([FLAGS.image_height, FLAGS.image_width, 2])
    x[:,:,0] = s
    x[ph,pw,1] = 1.0

    # # debug
    # plt.subplot(131)
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.subplot(132)
    # x_ = np.concatenate((x, np.zeros([FLAGS.image_height, FLAGS.image_width, 1])), axis=2)
    # plt.imshow(x_)
    # plt.subplot(133)
    # plt.imshow(y[:,:,0], cmap=plt.cm.gray)
    # plt.show()

    return x, y


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('pathnet'):
        working_path = os.path.join(current_path, 'vectornet/pathnet')
        os.chdir(working_path)

    # parameters 
    flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_threads = 8
    FLAGS.min_prop = 0.01

    # test
    # for _ in xrange(10):
    x_, y_ = preprocess(os.path.join(FLAGS.data_dir, 'archi.svg_pre'), FLAGS)


    batch_manager = BatchManager()
    x, y = batch_manager.batch()

    sess = tf.Session()
    batch_manager.start_thread(sess)

    test_iter = 1
    start_time = time.time()
    for _ in xrange(test_iter):
        x_batch, y_batch = sess.run([x, y])
    duration = time.time() - start_time
    duration = duration / test_iter
    batch_manager.stop_thread()
    print ('%s: %.3f sec/batch' % (datetime.now(), duration))

    x_batch = np.concatenate((x_batch, np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1])), axis=3)        
    plt.figure()
    for i in xrange(FLAGS.batch_size):
        x = np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width, 3])
        y = np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width])
        plt.subplot(121)
        plt.imshow(x)
        plt.subplot(122)
        plt.imshow(y, cmap=plt.cm.gray)
        plt.show()

    print('Done')
