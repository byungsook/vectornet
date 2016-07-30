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
tf.app.flags.DEFINE_float('intensity_ratio', 10.0,
                          """intensity ratio of point to lines""")
    

class LinenetManager(object):
    """
    Linenet
    """
    def __init__(self, img_shape, ckpt_path):
        self._graph = tf.Graph()
        with self._graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')

            self._x = tf.placeholder(dtype=tf.float32, shape=[None, img_shape[0], img_shape[1], 1])
            self._y_hat = linenet.linenet_model.inference(self._x, None, self._phase_train)

            self._sess = tf.Session()
        
            saver = tf.train.Saver()
            saver.restore(self._sess, ckpt_path)
            print('%s: Pre-trained model restored from %s' % (datetime.now(), ckpt_path))


    def extract_line(self, x, px, py):
        x = x / FLAGS.intensity_ratio
        x[px, py] = 1.0 # 0.2 for debug
        
        # # debug
        scipy.misc.imsave('./tmp.png', x)
        # plt.imshow(x, cmap=plt.cm.gray)
        # plt.show()
        
        x_shape = x.shape
        x = np.reshape(x, [1, x_shape[0], x_shape[1], 1])

        with self._graph.as_default():
            y = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x})
            y = np.reshape(y, [x_shape[0], x_shape[1]])
            
            # # debug
            # plt.imshow(y, cmap=plt.cm.gray)
            # plt.show()        
            
            return y
        

        