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

import tensorflow as tf


# parameters
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 8,
                     """Number of images to process in a batch.""")
flags.DEFINE_string('data_dir', '../data/bicycle',
                    """Path to the chinese data directory.""")
flags.DEFINE_integer('image_width', 128,
                     """Image Width.""")
flags.DEFINE_integer('image_height', 96,
                     """Image Height.""")
flags.DEFINE_integer('num_threads', 8, 
                     """# of threads for batch generation.""")
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

        FLAGS.num_threads = np.amin([multiprocessing.cpu_count(), FLAGS.batch_size])


        input_shape = [FLAGS.image_height, FLAGS.image_width, 1]

        self._q = tf.FIFOQueue(FLAGS.batch_size*10, [tf.float32, tf.float32], 
                               shapes=[input_shape, input_shape])

        self._x = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self._y = tf.placeholder(dtype=tf.float32, shape=input_shape)
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
        svg = sf.read().format(w=FLAGS.image_width, h=FLAGS.image_height,
                               r=0, sx=1, sy=1, tx=0, ty=0)

    x_png = cairosvg.svg2png(bytestring=svg)
    x_img = Image.open(io.BytesIO(x_png))
    x = np.array(x_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(x)
    x = x / max_intensity

    # # debug
    # plt.imshow(x, cmap=plt.cm.gray)
    # plt.show()


    y = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.bool)
    stroke_list = []
    svg_xml = et.fromstring(svg)
    num_paths = len(svg_xml[0])

    for i in xrange(1,num_paths):
        svg_xml = et.fromstring(svg)
        stroke = svg_xml[0][i]
        for c in reversed(xrange(1,num_paths)):
            if svg_xml[0][c] != stroke:
                svg_xml[0].remove(svg_xml[0][c])
        svg_one_stroke = et.tostring(svg_xml, method='xml')

        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke = (np.array(stroke_img)[:,:,3] > 0)

        # # debug
        # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
        # plt.imshow(stroke_img, cmap=plt.cm.gray)
        # plt.show()

        stroke_list.append(stroke)

    num_paths = len(stroke_list)
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

    # # debug
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(x, cmap=plt.cm.gray)
    # plt.subplot(122)
    # plt.imshow(y, cmap=plt.cm.gray)
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()

    x = np.reshape(x, [FLAGS.image_height, FLAGS.image_width, 1])
    y = np.reshape(y, [FLAGS.image_height, FLAGS.image_width, 1])
    return x, y


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    print(current_path)
    if not current_path.endswith('ovnet'):
        working_path = os.path.join(current_path, 'ovnet')
        os.chdir(working_path)

    # parameters 
    tf.app.flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_threads = 8

    # test
    x_, y_ = preprocess(os.path.join(FLAGS.data_dir, 'n02834778_63-1.svg_pre'), FLAGS)

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

    plt.figure()
    for i in xrange(FLAGS.batch_size):
        x = np.reshape(x_batch[i,:], [FLAGS.image_height, FLAGS.image_width])
        y = np.reshape(y_batch[i,:], [FLAGS.image_height, FLAGS.image_width])
        plt.subplot(121)
        plt.imshow(x, cmap=plt.cm.gray)
        plt.subplot(122)
        plt.imshow(y, cmap=plt.cm.gray)
        plt.show()

    print('Done')
