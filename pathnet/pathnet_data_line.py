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
flags.DEFINE_integer('batch_size', 16,
                     """Number of images to process in a batch.""")
flags.DEFINE_integer('image_width', 128,
                     """Image Width.""")
flags.DEFINE_integer('image_height', 96,
                     """Image Height.""")
flags.DEFINE_integer('num_threads', 8,
                     """# of threads for batch generation.""")
flags.DEFINE_integer('min_length', 10,
                     """minimum length of a line.""")
flags.DEFINE_integer('num_paths', 10,
                     """# paths for batch generation""")
flags.DEFINE_integer('path_type', 2,
                     """path type 0:line, 1:curve, 2:all""")
flags.DEFINE_integer('max_stroke_width', 2,
                     """max stroke width""")
flags.DEFINE_boolean('use_two_channels', True,
                     """use two channels for input""")
FLAGS = flags.FLAGS


SVG_START_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none">"""
SVG_LINE_TEMPLATE = """<path id="{id}" d="M {x1} {y1} L{x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" />"""
SVG_CUBIC_BEZIER_TEMPLATE = """<path id="{id}" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}" stroke="rgb({r},{g},{b})" stroke-width="{sw}" />"""
SVG_END_TEMPLATE = """</g></svg>"""


def _create_a_line(id, image_height, image_width, min_length, max_stroke_width):
    stroke_color = np.random.randint(240, size=3)
    # stroke_width = np.random.rand() * max_stroke_width + 2
    stroke_width = max_stroke_width
    while True:
        x = np.random.randint(low=0, high=image_width, size=2)
        y = np.random.randint(low=0, high=image_height, size=2)
        if x[0] - x[1] + y[0] - y[1] < min_length:
            continue
        break

    return SVG_LINE_TEMPLATE.format(
        id=id,
        x1=x[0], y1=y[0],
        x2=x[1], y2=y[1],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2], 
        sw=stroke_width
    )


def _create_a_cubic_bezier_curve(id, image_height, image_width, min_length, max_stroke_width):
    x = np.random.randint(low=0, high=image_width, size=4)
    y = np.random.randint(low=0, high=image_height, size=4)
    stroke_color = np.random.randint(240, size=3)
    # stroke_width = np.random.rand() * max_stroke_width + 2
    stroke_width = max_stroke_width

    return SVG_CUBIC_BEZIER_TEMPLATE.format(
        id=id,
        sx=x[0], sy=y[0],
        cx1=x[1], cy1=y[1],
        cx2=x[2], cy2=y[2],
        tx=x[3], ty=y[3],
        r=stroke_color[0], g=stroke_color[1], b=stroke_color[2], 
        sw=stroke_width
    )


def _create_a_path(path_type, id, FLAGS):
    if path_type == 2:
        path_type = np.random.randint(2)

    path_selector = {
        0: _create_a_line,
        1: _create_a_cubic_bezier_curve
    }

    return path_selector[path_type](id, FLAGS.image_height, FLAGS.image_width, 
                                    FLAGS.min_length, FLAGS.max_stroke_width)

class BatchManager(object):
    def __init__(self):
        self.num_examples_per_epoch = 1000
        self.num_epoch = 1

        FLAGS.num_threads = np.amin([FLAGS.num_threads, multiprocessing.cpu_count(), FLAGS.batch_size])

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
        print('%s: start to enque with %d threads' % (datetime.now(), FLAGS.num_threads))

        # Main thread: create a coordinator.
        self._sess = sess
        self._coord = tf.train.Coordinator()

        # Create a method for loading and enqueuing
        def load_n_enqueue(sess, enqueue, coord, x, y, FLAGS):
            with coord.stop_on_exception():
                while not coord.should_stop():
                    x_, y_ = preprocess(FLAGS)
                    sess.run(enqueue, feed_dict={x: x_, y: y_})

        # Create threads that enqueue
        self._threads = [threading.Thread(target=load_n_enqueue, 
                                          args=(self._sess, 
                                                self._enqueue,
                                                self._coord,
                                                self._x,
                                                self._y,
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


def preprocess(FLAGS):
    while True:
        svg = SVG_START_TEMPLATE.format(
                    width=FLAGS.image_width,
                    height=FLAGS.image_height
                )

        num_paths = np.random.randint(FLAGS.num_paths) + 1
        path_id = np.random.randint(num_paths)
        for i in xrange(num_paths):
            LINE = _create_a_path(FLAGS.path_type, i, FLAGS)
            svg += LINE

            svg_one_stroke = SVG_START_TEMPLATE.format(
                    width=FLAGS.image_width,
                    height=FLAGS.image_height
                ) + LINE + SVG_END_TEMPLATE
            
            if i == path_id:
                y_png = cairosvg.svg2png(bytestring=svg_one_stroke.encode('utf-8'))
                y_img = Image.open(io.BytesIO(y_png))

        svg += SVG_END_TEMPLATE
        s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        
        if max_intensity == 0:
            continue
        else:
            s = s / max_intensity

        # # debug
        # plt.imshow(s_img)
        # plt.show()

        # leave only one path
        y = np.array(y_img)[:,:,3].astype(np.float) / max_intensity

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

    y = np.reshape(y, [FLAGS.image_height, FLAGS.image_width, 1])
    x = np.zeros([FLAGS.image_height, FLAGS.image_width, 2])
    x[:,:,0] = s
    x[px,py,1] = 1.0

    return x, y


if __name__ == '__main__':
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('pathnet'):
        working_path = os.path.join(current_path, 'vectornet/pathnet')
        os.chdir(working_path)

    # parameters 
    FLAGS.num_threads = 8

    # # test
    # x_, y_ = preprocess(FLAGS)


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
