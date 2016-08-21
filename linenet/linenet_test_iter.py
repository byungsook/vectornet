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
tf.app.flags.DEFINE_string('test_dir', 'test/tmp',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', 'log/ratio_test/10_100000_30000_0.01/linenet.ckpt',
                           """If specified, restore this pretrained model.""")
tf.app.flags.DEFINE_string('train_model', 1,
			   """train model type [1-2] """)
tf.app.flags.DEFINE_integer('max_images', 3,
                            """max # images to save.""")
tf.app.flags.DEFINE_integer('num_eval', 6400,
                            """# images for evaluation""")
tf.app.flags.DEFINE_integer('num_line_ext', 10,
                            """# iteration for convergence of line extraction""")


def test_iter():
    with tf.Graph().as_default() as g:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        is_train = False
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        
        batch_manager = linenet_data.BatchManager()
        x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])
        y = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])
        x_no_p = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])

        # Build a Graph that computes the logits predictions from the inference model.
        y_hat = linenet_model.inference(x, phase_train, model=FLAGS.train_model)

        # Calculate loss.
        loss = linenet_model.loss(y_hat, y)

        saver = tf.train.Saver()


        # Build the summary writer
        summary_writer = [tf.train.SummaryWriter(FLAGS.test_dir + '/%02d' % i, g) for i in xrange(FLAGS.num_line_ext)]

        # summary at each iteration
        loss_ph = [tf.placeholder(tf.float32)] * FLAGS.num_line_ext
        loss_summary = [tf.scalar_summary('l2 loss (raw)', l) for l in loss_ph]
        
        loss_avg = tf.placeholder(tf.float32)
        loss_avg_summary = tf.scalar_summary('l2 loss (avg)', loss_avg)

	i_summary = tf.image_summary('I', x_no_p, max_images=FLAGS.max_images)
        x_summary = tf.image_summary('x', x, max_images=FLAGS.max_images)
        y_summary = tf.image_summary('y', y, max_images=FLAGS.max_images)
        y_hat_ph = tf.placeholder(tf.float32)
        y_hat_summary = tf.image_summary('y_hat_ph', y_hat_ph, max_images=FLAGS.max_images)

        
        # Start evaluation
        with tf.Session() as sess:
            if FLAGS.pretrained_model_checkpoint_path:
                assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
                saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
            
            num_iter = int(math.ceil(FLAGS.num_eval / FLAGS.batch_size))
            print('total iter: %d' % num_iter)
            total_loss = [0] * FLAGS.num_line_ext
            for step in xrange(num_iter):
                x_batch, y_batch, x_no_p_batch, p_batch = batch_manager.batch()
                # x_batch, y_batch, x_no_p_batch, p_batch = linenet_data.batch()
                for i in xrange(FLAGS.num_line_ext):
                    start_time = time.time()
                    y_hat_value, loss_value = sess.run([y_hat, loss], feed_dict={phase_train: is_train,
                                                                                x: x_batch, y: y_batch,
                                                                                x_no_p: x_no_p_batch})
                    total_loss[i] += loss_value
                    duration = time.time() - start_time
                    examples_per_sec = FLAGS.batch_size / float(duration)
                    print('%s: step %d-%d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
                            datetime.now(), step, i, loss_value, examples_per_sec, duration))

                    loss_summary_str = sess.run(loss_summary[i], feed_dict={loss_ph[i]: loss_value})
                    summary_writer[i].add_summary(loss_summary_str, step)

                    if step == 0:
                        i_summary_str, x_summary_str, y_summary_str, y_hat_summary_str = sess.run([i_summary, x_summary, y_summary, y_hat_summary], 
                            feed_dict={x_no_p: x_no_p_batch, x: x_batch, y: y_batch, y_hat_ph: y_hat_value})
			summary_writer[i].add_summary(i_summary_str, step)
                        summary_writer[i].add_summary(x_summary_str, step)
                        summary_writer[i].add_summary(y_summary_str, step)
                        summary_writer[i].add_summary(y_hat_summary_str, step)

                    # generate new x_batch from y_hat with p_batch
                    linenet_data.new_x_from_y_with_p(x_batch, y_hat_value, p_batch)

            for i in xrange(FLAGS.num_line_ext):
                # Compute precision
                loss_avg_ = total_loss[i] / num_iter / FLAGS.num_eval
                print('%s: %d, loss avg = %.7f' % (datetime.now(), i, loss_avg_))

                loss_avg_summary_str = sess.run(loss_avg_summary, feed_dict={loss_avg: loss_avg_})
                summary_writer[0].add_summary(loss_avg_summary_str, i)

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


    test_iter()
    # test_intersect()
    

if __name__ == '__main__':
    tf.app.run()  
