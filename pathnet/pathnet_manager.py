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

import pathnet.pathnet_model


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pathnet_ckpt', 'pathnet/model/no_trans_64/ch1/pathnet.ckpt-50000',
                           """pathnet checkpoint file path.""")
tf.app.flags.DEFINE_boolean('use_two_channels', True,
                            """use two channels for input""")


class PathnetManager(object):
    """
    Pathnet
    """
    def __init__(self, img_shape, crop_size=-1):
        self._h = img_shape[0]
        self._w = img_shape[1]
        self.crop_size = crop_size
        self._graph = tf.Graph()
        with self._graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self._phase_train = tf.placeholder(tf.bool, name='phase_train')

            d = 2 if FLAGS.use_two_channels else 1
            if self.crop_size == -1:
                self._x = tf.placeholder(dtype=tf.float32, shape=[None, self._h, self._w, d])
            else:
                self._x = tf.placeholder(dtype=tf.float32, shape=[None, self.crop_size, self.crop_size, d])
            self._y_hat = pathnet.pathnet_model.inference(self._x, self._phase_train)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self._sess = tf.Session(config=config)
        
            saver = tf.train.Saver()
            saver.restore(self._sess, FLAGS.pathnet_ckpt)
            print('%s: Pre-trained model restored from %s' % (datetime.now(), FLAGS.pathnet_ckpt))


    def extract_all(self, img, batch_size=None):
        """extract paths from all path pixels

        Args:
            img: Input image. 2D Tensor of [image_size, image_size]  
        Returns:
            y: 3D Tensor of [# path pixels, image_size, image_size]
            path_pixels: coordinates of all path pixels
        """

        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0]) 
        assert(num_path_pixels > 0)

        if batch_size is None:            
            if FLAGS.use_two_channels:
                x_batch = np.zeros([num_path_pixels, self._h, self._w, 2])
                for i in xrange(num_path_pixels):
                    x_batch[i,:,:,0] = img
                    px, py = path_pixels[0][i], path_pixels[1][i]
                    x_batch[i,px,py,1] = 1.0
            else:
                img = img / FLAGS.intensity_ratio

                x_batch = np.zeros([num_path_pixels, self._h, self._w])
                for i in xrange(num_path_pixels):
                    px, py = path_pixels[0][i], path_pixels[1][i]
                    x_batch[i,:,:] = img
                    x_batch[i,px,py] = 1.0

                    # # debug
                    # plt.imshow(x_batch[i,:,:], cmap=plt.cm.gray)
                    # plt.show()
                
                x_batch = np.reshape(x_batch, [num_path_pixels, self._h, self._w, 1])
            
            with self._graph.as_default():
                y_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
                
                # # debug
                # y_vis = np.reshape(y_batch[0,:,:,:], [self._h, self._w])
                # plt.imshow(y_vis, cmap=plt.cm.gray)
                # plt.show()
                
                return y_batch, path_pixels
        else:
            y_batch = None
            for b in xrange(0,num_path_pixels,batch_size):
                b_size = min(batch_size, num_path_pixels - b)
                x_batch = np.zeros([b_size, self._h, self._w, 2])
                for i in xrange(b_size):
                    x_batch[i,:,:,0] = img
                    px, py = path_pixels[0][b+i], path_pixels[1][b+i]
                    x_batch[i,px,py,1] = 1.0
            
                with self._graph.as_default():
                    y_b = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
                    if y_batch is None:
                        y_batch = y_b
                    else:
                        y_batch = np.concatenate((y_batch, y_b), axis=0)

            return y_batch, path_pixels

    def extract(self, img, px, py):
        """extract paths from px, py"""

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


    def extract_crop(self, img, batch_size):
        """extract by cropping"""

        dist = center = int((self.crop_size - 1) / 2)

        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0]) 
        assert(num_path_pixels > 0)

        id_start = 0
        id_end = min(batch_size, num_path_pixels)

        y_batch = None
        while True:
            bs = min(batch_size, id_end - id_start)

            x_batch = np.zeros([batch_size, self.crop_size, self.crop_size, 2])
            x_batch[:,center,center,1] = 1.0
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
                    x_batch[i,bx_start:bx_end,by_start:by_end,0] = img[cx_start:cx_end, cy_start:cy_end]
                except:
                    print(bx_start,bx_end,by_start,by_end)
                    print(cx_start,cx_end,cy_start,cy_end)
                    
            with self._graph.as_default():
                y_batch_ = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
                # # debug
                # y_vis = np.reshape(y_batch[0,:,:,:], [self.crop_size, self.crop_size])
                # plt.imshow(y_vis, cmap=plt.cm.gray)
                # plt.show()

            if y_batch is None:
                y_batch = y_batch_
            else:
                y_batch = np.concatenate((y_batch, y_batch_), axis=0)
            
            if id_end == num_path_pixels:
                break
            else:
                id_start = id_end
                id_end = min(id_end + batch_size, num_path_pixels)

        return y_batch, path_pixels, center


    def extract_save(self, img, batch_size, save_path):
        """extract and save"""

        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0]) 
        assert(num_path_pixels > 0)

        if not FLAGS.use_two_channels:
            img = img / FLAGS.intensity_ratio
        
        id_start = 0
        id_end = min(batch_size, num_path_pixels)
        while True:
            bs = min(batch_size, id_end - id_start)

            if FLAGS.use_two_channels:
                x_batch = np.zeros([batch_size, self._h, self._w, 2])
                for i in xrange(bs):
                    x_batch[i,:,:,0] = img
                    px, py = path_pixels[0][id_start+i], path_pixels[1][id_start+i]
                    x_batch[i,px,py,1] = 1.0
            else:
                x_batch = np.zeros([batch_size, self._h, self._w, 1])
                for i in xrange(bs):
                    px, py = path_pixels[0][id_start+i], path_pixels[1][id_start+i]
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

            if id_end == num_path_pixels:
                break
            else:
                id_start = id_end
                id_end = min(id_end + batch_size, num_path_pixels)


    def extract_save_crop(self, img, batch_size, save_path):
        """extract and save"""

        dist = center = int((self.crop_size - 1) / 2)

        path_pixels = np.nonzero(img)
        num_path_pixels = len(path_pixels[0]) 
        assert(num_path_pixels > 0)

        if not FLAGS.use_two_channels:
            img = img / FLAGS.intensity_ratio
        
        id_start = 0
        id_end = min(batch_size, num_path_pixels)
        while True:
            bs = min(batch_size, id_end - id_start)

            if FLAGS.use_two_channels:
                x_batch = np.zeros([batch_size, self.crop_size, self.crop_size, 2])
                x_batch[:,center,center,1] = 1.0
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
                    x_batch[i,bx_start:bx_end,by_start:by_end,0] = img[cx_start:cx_end, cy_start:cy_end]
            else:
                assert(False)
                # x_batch = np.zeros([batch_size, self._h, self._w, 1])
                # for i in xrange(bs):
                #     px, py = path_pixels[0][id_start+i], path_pixels[1][id_start+i]
                #     x_batch[i,:,:,0] = img
                #     x_batch[i,px,py,0] = 1.0

                    
            with self._graph.as_default():
                y_batch = self._sess.run(self._y_hat, feed_dict={self._phase_train: False, self._x: x_batch})
            
                # # debug
                # y_vis = np.reshape(y_batch[0,:,:,:], [self.crop_size, self.crop_size])
                # plt.imshow(y_vis, cmap=plt.cm.gray)
                # plt.show()


            # save
            for i in xrange(bs):
                y_vis = np.reshape(y_batch[i,:,:,:], [self.crop_size, self.crop_size])
                np.save(save_path.format(id=id_start+i), y_vis)

            if id_end == num_path_pixels:
                break
            else:
                id_start = id_end
                id_end = min(id_end + batch_size, num_path_pixels)
                