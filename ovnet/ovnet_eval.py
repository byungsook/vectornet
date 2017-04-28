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
from scipy.stats import threshold
import tensorflow as tf

import ovnet_model

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 'eval/test',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'log/ch1',
                           """If specified, restore this pretrained model.""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_integer('max_images', 8,
                            """max # images to save.""")
tf.app.flags.DEFINE_string('train_on', 'chinese',
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
    import ovnet_data_chinese
elif FLAGS.train_on == 'sketch':
    import ovnet_data_sketch
elif FLAGS.train_on == 'sketch2':
    import ovnet_data_sketch2
elif FLAGS.train_on == 'hand':
    import ovnet_data_hand
elif FLAGS.train_on == 'line':
    import ovnet_data_line
else:
    print('wrong training data set')
    assert(False)


def evaluate():
    with tf.Graph().as_default() as g:
        if FLAGS.train_on == 'chinese':
            batch_manager = ovnet_data_chinese.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'sketch':
            batch_manager = ovnet_data_sketch.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'sketch2':
            batch_manager = ovnet_data_sketch2.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'hand':
            batch_manager = ovnet_data_hand.BatchManager()
            print('%s: %d svg files' % (datetime.now(), batch_manager.num_examples_per_epoch))
        elif FLAGS.train_on == 'line':
            batch_manager = ovnet_data_line.BatchManager()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        is_train = False
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        # Build a Graph that computes the logits predictions from the inference model.
        x, y = batch_manager.batch()
        y_hat = ovnet_model.inference(x, phase_train)

        # Calculate loss.
        loss = ovnet_model.loss(y_hat, y)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # saver = tf.train.Saver()


        # Build the summary writer
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        acc_ph = tf.placeholder(tf.float32)
        acc_summary = tf.summary.scalar('IoU accuracy (raw)', acc_ph)

        acc_avg_ph = tf.placeholder(tf.float32)
        acc_avg_summary = tf.summary.scalar('IoU accuracy (avg)', acc_avg_ph)

        summary_y_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/y', g)
        summary_y_hat_writer = tf.summary.FileWriter(FLAGS.eval_dir + '/y_hat', g)

        x_summary = tf.summary.image('x', x, max_outputs=FLAGS.max_images)
        y_summary = tf.summary.image('y', y, max_outputs=FLAGS.max_images)
        y_hat_summary = tf.summary.image('y_hat', y_hat, max_outputs=FLAGS.max_images)

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
            print('total iter: %d' % num_iter)
            total_acc = 0
            for step in range(num_iter):
                start_time = time.time()
                loss_value = sess.run([loss, y_hat], feed_dict={phase_train: is_train})
                acc = 1.0 - loss_value                
                total_acc += acc
                duration = time.time() - start_time
                examples_per_sec = FLAGS.batch_size / float(duration)
                batch_manager.num_epoch = epoch_per_step*step
                print('%s:[epoch %.2f][step %d/%d] acc = %.2f (%.3f sec/batch)' % 
                      (datetime.now(), batch_manager.num_epoch, step, num_iter, acc, duration))

                acc_summary_str, x_summary_str, y_summary_str, y_hat_summary_str = sess.run(
                    [acc_summary, x_summary, y_summary, y_hat_summary],
                    feed_dict={acc_ph: acc})
                summary_writer.add_summary(acc_summary_str, step)

                x_summary_tmp = tf.Summary()
                y_summary_tmp = tf.Summary()
                y_hat_summary_tmp = tf.Summary()
                x_summary_tmp.ParseFromString(x_summary_str)
                y_summary_tmp.ParseFromString(y_summary_str)
                y_hat_summary_tmp.ParseFromString(y_hat_summary_str)
                for i in xrange(FLAGS.max_images):
                    new_tag = '%06d/%03d' % (step, i)
                    x_summary_tmp.value[i].tag = new_tag
                    y_summary_tmp.value[i].tag = new_tag
                    y_hat_summary_tmp.value[i].tag = new_tag

                summary_writer.add_summary(x_summary_tmp, step)
                summary_y_writer.add_summary(y_summary_tmp, step)
                summary_y_hat_writer.add_summary(y_hat_summary_tmp, step)

            batch_manager.stop_thread()
            
            # Compute precision
            acc_avg = total_acc / num_iter
            print('%s: IoU accuracy avg %.3f' % (datetime.now(), acc_avg))

            acc_avg_summary_str = sess.run(acc_avg_summary, feed_dict={acc_avg_ph: acc_avg})
            g_step = tf.train.global_step(sess, global_step)
            summary_writer.add_summary(acc_avg_summary_str, g_step)

    print('done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('ovnet'):
        working_path = os.path.join(current_path, 'vectornet/ovnet')
        os.chdir(working_path)
        
    # create eval directory
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    # start evaluation
    evaluate()

if __name__ == '__main__':
    tf.app.run()
