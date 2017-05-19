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
flags.DEFINE_integer('batch_size', 16,
                     """Number of images to process in a batch.""")
flags.DEFINE_string('data_dir', '../data/qdraw_baseball_128_all',
                    """Path to the Sketch data directory.""")
flags.DEFINE_integer('image_width', 128,
                     """Image Width.""")
flags.DEFINE_integer('image_height', 128,
                     """Image Height.""")
flags.DEFINE_integer('num_threads', 16,
                     """# of threads for batch generation.""")
FLAGS = flags.FLAGS


class BatchManager(object):
    def __init__(self):
        # read all svg files
        self._next_svg_id = 0
        self.data_list = []
        
        if FLAGS.file_list:
            file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
            with open(file_list_path, 'r') as f:
                while True:
                    line = f.readline()
                    if not line: break

                    file_path = os.path.join(FLAGS.data_dir, line.rstrip())
                    self.data_list.append(file_path)
        else:
            for root, _, files in os.walk(FLAGS.data_dir):
                for file in files:
                    if not file.lower().endswith('svg_pre'):
                        continue

                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        svg = f.read()
                        self._data_list.append(svg)

        self.num_examples_per_epoch = len(self.data_list)
        self.num_epoch = 1

        FLAGS.num_threads = np.amin([FLAGS.num_threads, multiprocessing.cpu_count()*2])


        image_shape = [FLAGS.image_height, FLAGS.image_width, 1]

        self._q = tf.FIFOQueue(FLAGS.batch_size*10, [tf.float32, tf.float32], 
                               shapes=[image_shape, image_shape])

        self._x = tf.placeholder(dtype=tf.float32, shape=image_shape)
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
        self._sess = sess
        self._coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, x, y, data_list, FLAGS):
            with coord.stop_on_exception():
                while not coord.should_stop():
                    svg_path = random.choice(data_list)
                    x_, y_ = preprocess(svg_path, FLAGS)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self._threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(self._sess, 
                                                self._enqueue,
                                                self._coord,
                                                self._x,
                                                self._y,
                                                self.data_list,
                                                FLAGS)
                                          ) for i in xrange(FLAGS.num_threads)]

        # define signal handler
        def signal_handler(signum,frame):
            #print "stop training, save checkpoint..."
            #saver.save(sess, "./checkpoints/VDSR_norm_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
            print('%s: canceled by SIGINT' % datetime.now())
            self._coord.request_stop()
            self._sess.run(self._q.close(cancel_pending_enqueues=True))
            self._coord.join(self._threads)
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the threads and wait for all of them to stop.
        for t in self._threads:
            t.start()

    def stop_thread(self):
        self._coord.request_stop()
        self._sess.run(self._q.close(cancel_pending_enqueues=True))
        self._coord.join(self._threads)


def preprocess(file_path, FLAGS):
    with open(file_path, 'r') as sf:
        svg = sf.read()

    x_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    x_img = Image.open(io.BytesIO(x_png))
    x = np.array(x_img)[:,:,3].astype(np.float)
    max_intensity = np.amax(x)
    x /= max_intensity
    if max_intensity == 0:
        y = np.zeros([FLAGS.image_height, FLAGS.image_width, 1])
        x = np.zeros([FLAGS.image_height, FLAGS.image_width, 1])
        return x, y

    # # debug
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.show()

    y = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.bool)
    stroke_list = []
    num_paths = svg.count('polyline')

    for i in xrange(1,num_paths+1):
        svg_xml = et.fromstring(svg)
        # svg_xml[0]._children = [svg_xml[0]._children[i]]
        stroke = svg_xml[i]
        for c in reversed(xrange(1,num_paths+1)):
            if svg_xml[c] != stroke:
                svg_xml.remove(svg_xml[c])
        svg_one_stroke = et.tostring(svg_xml, method='xml')

        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke = (np.array(stroke_img)[:,:,3] > 0)

        # # debug
        # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
        # plt.imshow(stroke_img, cmap=plt.cm.gray)
        # plt.show()

        stroke_list.append(stroke)

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
    if not current_path.endswith('pathnet'):
        working_path = os.path.join(current_path, 'vectornet/pathnet')
        os.chdir(working_path)

    # parameters 
    flags.DEFINE_string('file_list', 'train.txt', """file_list""")
    FLAGS.num_threads = 16

    # # test preprocess()
    # x, y = preprocess(os.path.join(FLAGS.data_dir, '4557720147460096.svg'), FLAGS)
    
    batch_manager = BatchManager()
    x, y = batch_manager.batch()

    sess = tf.Session()
    batch_manager.start_thread(sess)

    test_iter = 10
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
