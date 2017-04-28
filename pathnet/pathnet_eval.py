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

import pathnet_model

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'eval/test',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'model/ch1',
                           """If specified, restore this pretrained model.""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_string('train_on', 'sketch',
                           """specify training data""")
tf.app.flags.DEFINE_boolean('transform', False,
                            """Whether to transform character.""")
tf.app.flags.DEFINE_string('file_list', 'test.txt',
                           """file_list""")
tf.app.flags.DEFINE_integer('num_epoch', 10,
                            """# epoch""")
tf.app.flags.DEFINE_float('min_prop', 0.0,
                          """min_prop""")

if FLAGS.train_on == 'chinese':
    import pathnet_data_chinese
elif FLAGS.train_on == 'sketch':
    import pathnet_data_sketch
elif FLAGS.train_on == 'sketch2':
    import pathnet_data_sketch2
elif FLAGS.train_on == 'hand':
    import pathnet_data_hand
elif FLAGS.train_on == 'line':
    import pathnet_data_line
else:
    print('wrong training data set')
    assert(False)


def evaluate():
    with tf.Graph().as_default() as g:
        if FLAGS.train_on == 'chinese':
            batch_manager = pathnet_data_chinese.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'sketch':
            batch_manager = pathnet_data_sketch.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'sketch2':
            batch_manager = pathnet_data_sketch2.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'hand':
            batch_manager = pathnet_data_hand.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'line':
            batch_manager = pathnet_data_line.BatchManager()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        is_train = False
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        # Build a Graph that computes the logits predictions from the inference model.
        x, y = batch_manager.batch()
        y_hat = pathnet_model.inference(x, phase_train)

        # Calculate loss.
        loss = pathnet_model.loss(y_hat, y)

        # # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()


        # Build the summary writer
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        loss_ph = tf.placeholder(tf.float32)
        loss_summary = tf.summary.scalar('l2 loss (raw)', loss_ph)
        
        loss_avg = tf.placeholder(tf.float32)
        loss_avg_summary = tf.summary.scalar('l2 loss (avg)', loss_avg)

        summary_x_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/x', g)
        summary_y_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/y', g)
        summary_y_hat_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/y_hat', g)
        
        if FLAGS.use_two_channels:
            x_rgb = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 3])
            x_summary = tf.summary.image('x', x_rgb, max_outputs=FLAGS.max_images)
            b_channel = np.zeros([FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 1]) # to make x RGB
        y_img = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_hat_img = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, 1])
        y_summary = tf.summary.image('y', y_img, max_outputs=FLAGS.max_images)
        y_hat_summary = tf.summary.image('y_hat', y_hat_img, max_outputs=FLAGS.max_images)
        
        # Start evaluation
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and FLAGS.checkpoint_dir:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
                print('%s: Pre-trained model restored from %s' % 
                    (datetime.now(), ckpt_name))
            
            batch_manager.start_thread(sess)
            epoch_per_step = float(FLAGS.batch_size) / batch_manager.num_examples_per_epoch
            num_eval = batch_manager.num_examples_per_epoch * FLAGS.num_epoch
            num_iter = int(math.ceil(num_eval / FLAGS.batch_size))
            # num_iter = 1
            print('total iter: %d' % num_iter)
            total_loss = 0
            for step in range(num_iter):
                start_time = time.time()
                x_batch, y_batch, y_hat_batch, loss_value = sess.run([x, y, y_hat, loss],
                                                                     feed_dict={phase_train: is_train})
                total_loss += loss_value
                duration = time.time() - start_time
                batch_manager.num_epoch = epoch_per_step*step
                print('%s:[epoch %.2f][step %d/%d] loss = %.2f (%.3f sec/batch)' % 
                      (datetime.now(), batch_manager.num_epoch, step, num_iter, loss_value, duration))

                if FLAGS.use_two_channels:
                    loss_summary_str, x_summary_str, y_summary_str, y_hat_summary_str = sess.run(
                        [loss_summary, x_summary, y_summary, y_hat_summary],
                        feed_dict={phase_train: is_train, x_rgb: np.concatenate((x_batch, b_channel), axis=3),
                                   y_img: y_batch, y_hat_img: y_hat_batch})
                summary_writer.add_summary(loss_summary_str, step)

                x_summary_tmp = tf.Summary()
                y_summary_tmp = tf.Summary()
                y_hat_summary_tmp = tf.Summary()
                x_summary_tmp.ParseFromString(x_summary_str)
                y_summary_tmp.ParseFromString(y_summary_str)
                y_hat_summary_tmp.ParseFromString(y_hat_summary_str)
                for i in xrange(FLAGS.batch_size):
                    new_tag = '%06d/%03d' % (step, i)
                    x_summary_tmp.value[i].tag = new_tag
                    y_summary_tmp.value[i].tag = new_tag
                    y_hat_summary_tmp.value[i].tag = new_tag

                summary_x_writer.add_summary(x_summary_tmp, step)
                summary_y_writer.add_summary(y_summary_tmp, step)
                summary_y_hat_writer.add_summary(y_hat_summary_tmp, step)

            batch_manager.stop_thread()

            # Compute precision
            loss_avg_ = total_loss / num_iter
            print('%s: loss avg = %.3f' % (datetime.now(), loss_avg_))

            loss_avg_summary_str = sess.run(loss_avg_summary, feed_dict={loss_avg: loss_avg_})
            g_step = tf.train.global_step(sess, global_step)
            summary_writer.add_summary(loss_avg_summary_str, g_step)    
    print('done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('pathnet'):
        working_path = os.path.join(current_path, 'vectornet/pathnet')
        os.chdir(working_path)

    # create eval directory
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # start evaluation
    evaluate()

if __name__ == '__main__':
    tf.app.run()
