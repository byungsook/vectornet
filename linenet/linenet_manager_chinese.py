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
tf.app.flags.DEFINE_string('linenet_ckpt', 'model/chinese/linenet.ckpt',
                           """linenet checkpoint file path.""")  
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")


class LinenetManager(object):
    """
    Linenet
    """
    def __init__(self, img_shape):
        self._h = img_shape[0]
        self._w = img_shape[1]
        self._graph = tf.Graph()
        with self._graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')

            d = 2 if FLAGS.use_two_channels else 1
            self._x = tf.placeholder(dtype=tf.float32, shape=[None, self._h, self._w, d])
            self._y_hat = linenet.linenet_model.inference(self._x, self._phase_train)

            self._sess = tf.Session()
        
            saver = tf.train.Saver()
            saver.restore(self._sess, FLAGS.linenet_ckpt)
            print('%s: Pre-trained model restored from %s' % (datetime.now(), FLAGS.linenet_ckpt))


    def extract_all(self, img):
        """extract lines from all line pixels

        Args:
            img: Input image. 2D Tensor of [image_size, image_size]  
        Returns:
            y: 3D Tensor of [# line pixels, image_size, image_size]
            line_pixels: coordinates of all line pixels
        """

        line_pixels = np.nonzero(img)
        num_line_pixels = len(line_pixels[0]) 
        assert(num_line_pixels > 0)
        
        if FLAGS.use_two_channels:
            x_batch = np.zeros([num_line_pixels, self._h, self._w, 2])
            for i in xrange(num_line_pixels):
                x_batch[i,:,:,0] = img
                px, py = line_pixels[0][i], line_pixels[1][i]
                x_batch[i,px,py,1] = 1.0
        else:
            img = img / FLAGS.intensity_ratio

            x_batch = np.zeros([num_line_pixels, self._h, self._w])
            for i in xrange(num_line_pixels):
                px, py = line_pixels[0][i], line_pixels[1][i]
                x_batch[i,:,:] = img
                x_batch[i,px,py] = 1.0

                # # debug
                # plt.imshow(x_batch[i,:,:], cmap=plt.cm.gray)
                # plt.show()
            
            x_batch = np.reshape(x_batch, [num_line_pixels, self._h, self._w, 1])
        
        with self._graph.as_default():
            y_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
            # # debug
            # y_vis = np.reshape(y_batch[0,:,:,:], [self._h, self._w])
            # plt.imshow(y_vis, cmap=plt.cm.gray)
            # plt.show()
            
            return y_batch, line_pixels

    def extract(self, img, px, py):
        """extract lines from px, py"""

        if FLAGS.use_two_channels:
            x_batch = np.zeros([1, self._h, self._w, 2])
            x_batch[0,:,:,0] = img
            x_batch[0,px,py,1] = 1.0
        else:
            img = img / FLAGS.intensity_ratio

            x_batch = np.zeros([1, self._h, self._w, 1])
            x_batch[i,:,:,0] = img
            x_batch[i,px,py,0] = 1.0

            # # debug
            # plt.imshow(x_batch[i,:,:,0], cmap=plt.cm.gray)
            # plt.show()
            
        with self._graph.as_default():
            y_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
            # # debug
            # y_vis = np.reshape(y_batch[0,:,:,:], [self._h, self._w])
            # plt.imshow(y_vis, cmap=plt.cm.gray)
            # plt.show()

        y = np.reshape(y_batch[0,:,:,0], [img.shape[0], img.shape[1]])
        return y


    def extract_save(self, img, batch_size, save_path):
        """extract and save"""

        line_pixels = np.nonzero(img)
        num_line_pixels = len(line_pixels[0]) 
        assert(num_line_pixels > 0)

        if not FLAGS.use_two_channels:
            img = img / FLAGS.intensity_ratio
        
        id_start = 0
        id_end = batch_size
        while True:
            bs = min(batch_size, id_end - id_start)

            if FLAGS.use_two_channels:
                x_batch = np.zeros([batch_size, self._h, self._w, 2])
                for i in xrange(bs):
                    x_batch[i,:,:,0] = img
                    px, py = line_pixels[0][id_start+i], line_pixels[1][id_start+i]
                    x_batch[i,px,py,1] = 1.0
            else:
                x_batch = np.zeros([batch_size, self._h, self._w, 1])
                for i in xrange(bs):
                    px, py = line_pixels[0][id_start+i], line_pixels[1][id_start+i]
                    x_batch[i,:,:,0] = img
                    x_batch[i,px,py,0] = 1.0

                    
            with self._graph.as_default():
                y_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
                # # debug
                # y_vis = np.reshape(y_batch[0,:,:,:], [self._h, self._w])
                # plt.imshow(y_vis, cmap=plt.cm.gray)
                # plt.show()


            # save
            for i in xrange(bs):
                y_vis = np.reshape(y_batch[i,:,:,:], [self._h, self._w])                
                np.save(save_path.format(id=id_start+i), y_vis)

            if id_end == num_line_pixels:
                break
            else:
                id_start = id_end
                id_end = min(id_end + batch_size, num_line_pixels)
                