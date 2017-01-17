# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc

import linenet.linenet_model


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('intersectnet_ckpt', 'model/overlap_line_train/linenet.ckpt',
                           """intersectnet checkpoint file path.""")


class IntersectnetManager(object):
    def __init__(self, img_shape):
        self._h = img_shape[0]
        self._w = img_shape[1]
        self._graph = tf.Graph()
        with self._graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')

            self._x = tf.placeholder(dtype=tf.float32, shape=[None, self._h, self._w, 1])
            self._y_hat = linenet.linenet_model.inference(self._x, self._phase_train)

            self._sess = tf.Session()

            saver = tf.train.Saver()
            saver.restore(self._sess, FLAGS.intersectnet_ckpt)
            print('%s: Pre-trained model restored from %s' % (datetime.now(), FLAGS.intersectnet_ckpt))


    def intersect(self, img):        
        x_batch = np.zeros([1, self._h, self._w, 1])
        x_batch[0,:,:,0] = img
        
        # # debug
        # plt.imshow(x_batch[0,:,:,0], cmap=plt.cm.gray)
        # plt.show()
        
        with self._graph.as_default():
            y_hat_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
            # # debug
            # y_vis = np.reshape(y_hat_batch[0,:,:,:], [self._h, self._w])
            # plt.imshow(y_vis, cmap=plt.cm.gray)
            # plt.show()

            return y_hat_batch