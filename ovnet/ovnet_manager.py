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
import scipy.stats

import ovnet.ovnet_model


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ovnet_ckpt', 'ovnet/model/no_trans_64/ch1/ovnet.ckpt-50000',
                           """pathnet checkpoint file path.""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_float('threshold', 0.5,
                          """threshold""")

class OvnetManager(object):
    """
    Ovnet
    """
    def __init__(self, img_shape, crop_size=-1):
        self._h = img_shape[0]
        self._w = img_shape[1]
        self._crop_size = crop_size
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')
            self._x = tf.placeholder(dtype=tf.float32, shape=[None, self._h, self._w, 1])
            self._y_hat = ovnet.ovnet_model.inference(self._x, self._phase_train)
            self._sess = tf.Session()
            
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # saver = tf.train.Saver()
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(self._sess, FLAGS.ovnet_ckpt)
            print('%s: Pre-trained model restored from %s' % (datetime.now(), FLAGS.ovnet_ckpt))


    def overlap(self, img):
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

            y = np.reshape(y_hat_batch[0,:,:,0], [img.shape[0], img.shape[1]])
            y = (scipy.stats.threshold(y, threshmin=FLAGS.threshold, newval=0) > 0)
            return y