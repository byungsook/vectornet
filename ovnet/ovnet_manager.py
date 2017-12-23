# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
from skimage import transform

import ovnet.ovnet_model


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ovnet_ckpt', 'ovnet/log/64/line/ovnet.ckpt-50000',
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
        self._h = img_shape[0] # 1024
        self._w = img_shape[1] # 1024
        self.crop_size = crop_size
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')
            if self.crop_size == -1:
                self._x = tf.placeholder(dtype=tf.float32, shape=[None, self._h, self._w, 1])
            else:
                self._x = tf.placeholder(dtype=tf.float32, shape=[None, self.crop_size, self.crop_size, 1])
            self._y_hat = ovnet.ovnet_model.inference(self._x, self._phase_train)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self._sess = tf.Session(config=config)
            
            global_step = tf.Variable(0, name='global_step', trainable=False)
            saver = tf.train.Saver()
            # variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
            # variables_to_restore = variable_averages.variables_to_restore()
            # saver = tf.train.Saver(variables_to_restore)
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

    def overlap_crop(self, img, batch_size):
        """extract by cropping"""

        dist = center = int((self.crop_size - 1) / 2)

        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0]) 
        assert(num_path_pixels > 0)

        id_start = 0
        id_end = min(batch_size, num_path_pixels)

        y = np.zeros([self._h, self._w])
        while True:
            bs = min(batch_size, id_end - id_start)

            x_batch = np.zeros([bs, self.crop_size, self.crop_size, 1])
            for i in xrange(bs):
                px, py = path_pixels[0][id_start+i], path_pixels[1][id_start+i]
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
                try:
                    img_crop = img[cx_start:cx_end, cy_start:cy_end]
                    max_intensity = np.amax(img_crop)
                    img_crop /= max_intensity
                    x_batch[i,bx_start:bx_end,by_start:by_end,0] = img_crop
                except:
                    print(bx_start,bx_end,by_start,by_end)
                    print(cx_start,cx_end,cy_start,cy_end)
                    
            with self._graph.as_default():
                y_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
                # # debug
                # x_vis1 = np.reshape(x_batch[0,:,:,:], [self.crop_size, self.crop_size])
                # y_vis1 = np.reshape(y_batch[0,:,:,:], [self.crop_size, self.crop_size])
                # x_vis2 = np.reshape(x_batch[-1,:,:,:], [self.crop_size, self.crop_size])
                # y_vis2 = np.reshape(y_batch[-1,:,:,:], [self.crop_size, self.crop_size])
                # print(np.amax(y_vis1), np.amax(y_vis2))
                # plt.figure()
                # plt.subplot(221)
                # plt.imshow(x_vis1, cmap=plt.cm.gray, clim=(0.0, 1.0))
                # plt.subplot(222)
                # plt.imshow(y_vis1, cmap=plt.cm.gray, clim=(0.0, 1.0))
                # plt.subplot(223)
                # plt.imshow(x_vis2, cmap=plt.cm.gray, clim=(0.0, 1.0))
                # plt.subplot(224)
                # plt.imshow(y_vis2, cmap=plt.cm.gray, clim=(0.0, 1.0))
                # plt.show()

                y_center = y_batch[:,center,center,0]
                y_overlap = np.where(y_center > FLAGS.threshold)[0]
                if len(y_overlap) > 0:
                    y[path_pixels[0][y_overlap+id_start], path_pixels[1][y_overlap+id_start]] = 1

            if id_end == num_path_pixels:
                break
            else:
                id_start = id_end
                id_end = min(id_end + batch_size, num_path_pixels)

        return y