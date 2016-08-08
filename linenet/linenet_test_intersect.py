# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import linenet_data
import linenet_model

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'test/intersect',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', 'log/ratio_test/10_100000_30000_0.01/linenet.ckpt',
                           """If specified, restore this pretrained model.""")


def test_intersect():
    with tf.Graph().as_default() as g:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        is_train = False
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        
        batch_manager = linenet_data.BatchManager()
        x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])
        y = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])
        x_no_p = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])

        # Build a Graph that computes the logits predictions from the inference model.
        y_hat = linenet_model.inference(x, x_no_p, phase_train)

        # Calculate loss.
        loss = linenet_model.loss(y_hat, y)

        saver = tf.train.Saver()


        # Build the summary writer
        summary_writer = tf.train.SummaryWriter(FLAGS.test_dir, g)

        loss_ph = tf.placeholder(tf.float32)
        loss_summary = tf.scalar_summary('l2 loss (raw)', loss_ph)
        
        summary_x_no_p_writer = tf.train.SummaryWriter(FLAGS.test_dir + '/x_no_p', g)
        summary_x_writer = tf.train.SummaryWriter(FLAGS.test_dir + '/x', g)
        summary_y_writer = tf.train.SummaryWriter(FLAGS.test_dir + '/y', g)
        summary_y_hat_writer = tf.train.SummaryWriter(FLAGS.test_dir + '/y_hat', g)

        num_images = 21
        x_no_p_summary = tf.image_summary('x_no_p', x_no_p, max_images=num_images)
        x_summary = tf.image_summary('x', x, max_images=num_images)
        y_summary = tf.image_summary('y', y, max_images=num_images)
        y_hat_ph = tf.placeholder(tf.float32)
        y_hat_summary = tf.image_summary('y_hat_ph', y_hat_ph, max_images=num_images)

        
        # Start evaluation
        with tf.Session() as sess:
            if FLAGS.pretrained_model_checkpoint_path:
                assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
                saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
            
            step = 0
            x_batch, y_batch, x_no_p_batch, _ = linenet_data.batch_for_intersection_test()        
            
            start_time = time.time()
            y_hat_value, loss_value = sess.run([y_hat, loss], feed_dict={phase_train: is_train,
                                                                            x: x_batch, y: y_batch,
                                                                            x_no_p: x_no_p_batch})
            duration = time.time() - start_time
            examples_per_sec = FLAGS.batch_size / float(duration)
            print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
                    datetime.now(), step, loss_value, examples_per_sec, duration))

            loss_summary_str, x_no_p_summary_str, x_summary_str, y_summary_str, y_hat_summary_str = sess.run(
                [loss_summary, x_no_p_summary, x_summary, y_summary, y_hat_summary],
                feed_dict={loss_ph: loss_value,
                            x_no_p: x_no_p_batch, x: x_batch, y: y_batch, y_hat_ph: y_hat_value})

            summary_writer.add_summary(loss_summary_str, step)

            x_no_p_summary_tmp = tf.Summary()
            x_summary_tmp = tf.Summary()
            y_summary_tmp = tf.Summary()
            y_hat_summary_tmp = tf.Summary()
            x_no_p_summary_tmp.ParseFromString(x_no_p_summary_str)
            x_summary_tmp.ParseFromString(x_summary_str)
            y_summary_tmp.ParseFromString(y_summary_str)
            y_hat_summary_tmp.ParseFromString(y_hat_summary_str)
            for i in xrange(FLAGS.max_images):
                new_tag = '%06d/%03d' % (step, i)
                x_no_p_summary_tmp.value[i].tag = new_tag
                x_summary_tmp.value[i].tag = new_tag
                y_summary_tmp.value[i].tag = new_tag
                y_hat_summary_tmp.value[i].tag = new_tag

            summary_x_no_p_writer.add_summary(x_no_p_summary_tmp, step)
            summary_x_writer.add_summary(x_summary_tmp, step)
            summary_y_writer.add_summary(y_summary_tmp, step)
            summary_y_hat_writer.add_summary(y_hat_summary_tmp, step)            

    print('done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('linenet'):
        working_path = os.path.join(current_path, 'vectornet/linenet')
        os.chdir(working_path)
        
    # create eval directory
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)

    
    test_intersect()
    

if __name__ == '__main__':
    tf.app.run()  
