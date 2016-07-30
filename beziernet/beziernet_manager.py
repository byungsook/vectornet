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
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf

import beziernet.beziernet_model


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('model', 1,
                            """model""")
tf.app.flags.DEFINE_integer('image_size', 96, # 96-48-24-12-6
                            """Image Size.""")


class BeziernetManager(object):
    """
    Beziernet Manager
    """
    def __init__(self, ckpt_path):
        self._graph = tf.Graph()
        with self._graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')

            self._x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])
            self._y_hat = beziernet.beziernet_model.inference(self._x, self._phase_train, model=FLAGS.model)

            self._sess = tf.Session()
        
            saver = tf.train.Saver()
            saver.restore(self._sess, ckpt_path)
            print('%s: Pre-trained model restored from %s' % (datetime.now(), ckpt_path))


    def fit_line(self, x):
        scale_x, scale_y = x.shape[0] / FLAGS.image_size, x.shape[1] / FLAGS.image_size
            
        x = scipy.misc.imresize(x, (FLAGS.image_size,FLAGS.image_size)) / 255.0
        
        # # debug
        # print(x.shape, np.amax(x))
        # plt.imshow(x, cmap=plt.cm.gray)
        # plt.show()

        x = np.reshape(x, [1, FLAGS.image_size, FLAGS.image_size, 1])
        with self._graph.as_default():
            y = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x})[0]
            
            y[0::2] *= scale_x
            y[1::2] *= scale_y
            
            return y
        

        