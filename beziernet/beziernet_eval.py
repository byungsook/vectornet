# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

"""Evaluation for Beziernet
"""
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

import beziernet_data
import beziernet_model

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'eval',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('use_train_data', False,
                           """Use either test or train data.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', 'log/second_train/beziernet.ckpt', # 'log/second_train/beziernet.ckpt',
                           """If specified, restore this pretrained model.""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_integer('max_images', FLAGS.batch_size,
                            """max # images to save.""")


def evaluate():
    with tf.Graph().as_default() as g:
        # Get images and labels for Beziernet.
        num_examples_for_evaluation = 0
        if FLAGS.use_train_data:
            num_examples_for_evaluation = FLAGS.num_examples_per_epoch_for_train
        else:
            num_examples_for_evaluation = FLAGS.num_examples_per_epoch_for_test

        global_step = tf.Variable(0, name='global_step', trainable=False)
        is_train = False
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        
        images, xys = beziernet_data.inputs(use_train_data=FLAGS.use_train_data, batch_shuffle=is_train)

        # Build a Graph that computes the logits predictions from the inference model.
        logits = beziernet_model.inference(images, phase_train)

        # Calculate loss.
        loss = beziernet_model.loss(logits, xys)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        x_summary = tf.image_summary('x', images, max_images=FLAGS.max_images)
        y_img = tf.placeholder(tf.uint8, shape=[FLAGS.max_images, FLAGS.image_size, FLAGS.image_size, 1])
        y_summary = tf.image_summary('y', y_img, max_images=FLAGS.max_images)
        precision_ = tf.placeholder(tf.float32)
        precision_summary = tf.scalar_summary('precision', precision_)

        with tf.Session() as sess:
            if FLAGS.pretrained_model_checkpoint_path:
                assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
                saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
                print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(num_examples_for_evaluation / FLAGS.batch_size))
                total_loss = 0  # Sum of losses
                total_sample_count = num_iter
                step = 0
                while step < num_iter and not coord.should_stop():
                    start_time = time.time()
                    loss_values = sess.run(loss, feed_dict={phase_train: is_train})
                    total_loss += loss_values
                    duration = time.time() - start_time
                    examples_per_sec = FLAGS.batch_size / float(duration)
                    print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
                            datetime.now(), step, loss_values, examples_per_sec, duration))                    

                    x_summary_str, y_eval = sess.run([x_summary, logits], # logits=xys
                                                    feed_dict={phase_train: is_train})
                    new_y_img = beziernet_data.svg_to_png(y_eval, FLAGS.max_images)
                    y_summary_str = sess.run(y_summary, feed_dict={y_img: new_y_img})

                    x_summary_tmp = tf.Summary()
                    y_summary_tmp = tf.Summary()
                    x_summary_tmp.ParseFromString(x_summary_str)
                    y_summary_tmp.ParseFromString(y_summary_str)
                    for i in xrange(FLAGS.max_images):
                        x_summary_tmp.value[i].tag = '%07d/%03d_x' % (step, i)
                        y_summary_tmp.value[i].tag = '%07d/%03d_y' % (step, i)
                    summary_writer.add_summary(x_summary_tmp, step)
                    summary_writer.add_summary(y_summary_tmp, step)

                    step += 1

                # Compute precision
                precision = total_loss / total_sample_count
                print('%s: precision = %.3f' % (datetime.now(), precision))

                precision_summary_str = sess.run(precision_summary, feed_dict={precision_: precision})
                g_step = tf.train.global_step(sess, global_step)
                summary_writer.add_summary(precision_summary_str, g_step)

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

    print('done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('beziernet'):
        working_path = os.path.join(current_path, 'vectornet/beziernet')
        os.chdir(working_path)
        
    # create eval directory
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # (optional) generate bezier bin data set or extract 
    # beziernet_data.generate_bezier_bin()
    # beziernet_data.extract_bezier_bin()

    # start evaluation
    evaluate()

if __name__ == '__main__':
    tf.app.run()
