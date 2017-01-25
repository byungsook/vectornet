# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc

import linenet.linenet_model


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('linenet_ckpt', 'model/line_train/linenet.ckpt',
                           """linenet checkpoint file path.""")
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")


class LinenetManager(object):
    """
    Linenet
    """
    def __init__(self, img_shape, crop_size=-1):
        self._h = img_shape[0]
        self._w = img_shape[1]
        self._crop_size = crop_size
        self._graph = tf.Graph()
        with self._graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')

            d = 2 if FLAGS.use_two_channels else 1
            if self._crop_size == -1:
                self._x = tf.placeholder(dtype=tf.float32, shape=[None, self._h, self._w, d])
            else:
                self._x = tf.placeholder(dtype=tf.float32, shape=[None, self._crop_size, self._crop_size, d])
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

    def extract_save(self, img, batch_size, save_path):
        if self._crop_size == -1:
            return self.extract_save_no_crop(img, batch_size, save_path)
        else:
            return self.extract_save_crop(img, batch_size, save_path)


    def extract_save_no_crop(self, img, batch_size, save_path):
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
        id_end = min(batch_size, num_line_pixels)
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


    def extract_save_crop(self, img, batch_size, save_path):
        """extract and save"""

        dist = center = int((self._crop_size - 1) / 2)

        line_pixels = np.nonzero(img)
        num_line_pixels = len(line_pixels[0]) 
        assert(num_line_pixels > 0)

        if not FLAGS.use_two_channels:
            img = img / FLAGS.intensity_ratio
        
        id_start = 0
        id_end = min(batch_size, num_line_pixels)
        while True:
            bs = min(batch_size, id_end - id_start)

            if FLAGS.use_two_channels:
                x_batch = np.zeros([batch_size, self._crop_size, self._crop_size, 2])
                x_batch[:,center,center,1] = 1.0
                for i in xrange(bs):
                    px, py = line_pixels[0][id_start+i], line_pixels[1][id_start+i]
                    cx_start = px - dist
                    cx_end = px + dist + 1
                    dx1 = dist
                    dx2 = dist + 1
                    if cx_start < 0:
                        cx_start = 0
                        dx1 = px - cx_start
                    elif cx_end >= self._h:
                        cx_end = self._h
                        dx2 = cx_end - px
                    
                    cy_start = py - dist
                    cy_end = py + dist + 1
                    dy1 = dist
                    dy2 = dist + 1
                    if cy_start < 0:
                        cy_start = 0
                        dy1 = py - cy_start
                    elif cy_end >= self._w:
                        cy_end = self._w
                        dy2 = cy_end - py
                    
                    bx_start = center - dx1
                    bx_end = center + dx2
                    by_start = center - dy1
                    by_end = center + dy2
                    x_batch[i,bx_start:bx_end,by_start:by_end,0] = img[cx_start:cx_end, cy_start:cy_end]
            else:
                assert(False)
                # x_batch = np.zeros([batch_size, self._h, self._w, 1])
                # for i in xrange(bs):
                #     px, py = line_pixels[0][id_start+i], line_pixels[1][id_start+i]
                #     x_batch[i,:,:,0] = img
                #     x_batch[i,px,py,0] = 1.0

                    
            with self._graph.as_default():
                y_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
                # # debug
                # y_vis = np.reshape(y_batch[0,:,:,:], [self._crop_size, self._crop_size])
                # plt.imshow(y_vis, cmap=plt.cm.gray)
                # plt.show()


            # save
            for i in xrange(bs):
                y_vis = np.reshape(y_batch[i,:,:,:], [self._crop_size, self._crop_size])
                np.save(save_path.format(id=id_start+i), y_vis)

            if id_end == num_line_pixels:
                break
            else:
                id_start = id_end
                id_end = min(id_end + batch_size, num_line_pixels)
                