# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

"""A binary to train Beziernet on the one-cubic-bezier-curve data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import tensorflow as tf

import beziernet_data
import beziernet_model

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', 'log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '', # 'log/second_train/beziernet.ckpt',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 237, # 90000/128= 703 steps/epoch, ~166k
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('learning_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.9999,
                          """The decay to use for the moving average.""")
tf.app.flags.DEFINE_integer('max_images', 1,
                            """max # images to save.""")


def train():
    """Train Vectornet for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        is_train = True
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        
        # Get images and xys for BezierNet.
        use_data = False
        if use_data:
            images, xys = beziernet_data.inputs()
        else:
            # custom_runner = beziernet_data.CustomRunner()
            # images, xys = custom_runner.inputs()
            images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1])
            xys = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.xy_size])


        # Build a Graph that computes the logits predictions from the inference model.
        logits = beziernet_model.inference(images, phase_train, model=3)

        # Calculate loss.
        loss = beziernet_model.loss(logits, xys)

        ###############################################################################
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply([loss])

        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(loss.op.name + ' (raw)', loss)
        tf.scalar_summary(loss.op.name, loss_averages.average(loss))
        
        # Variables that affect learning rate.
        num_batches_per_epoch = FLAGS.num_examples_per_epoch_for_train / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   FLAGS.learning_decay_factor,
                                                   staircase=True)
        tf.scalar_summary('learning_rate', learning_rate)
        
        # # or use fixed learning rate
        # learning_rate = 1e-3
        
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(learning_rate)
            # opt = tf.train.AdadeltaOptimizer()
            grads = opt.compute_gradients(loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        

        ####################################################################
        # Start running operations on the Graph. 
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(tf.initialize_all_variables())

        # Create a saver (restorer).
        saver = tf.train.Saver()
        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # Build the summary operation.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

        x_summary = tf.image_summary('x', images, max_images=FLAGS.max_images)
        y_img = tf.placeholder(tf.uint8, shape=[FLAGS.max_images, FLAGS.image_size, FLAGS.image_size, 1])
        y_summary = tf.image_summary('y', y_img, max_images=FLAGS.max_images)

        # # Start the queue runners.
        # tf.train.start_queue_runners(sess=sess)
        # if not use_data:
        #     # start our custom queue runner's threads
        #     custom_runner.start_threads(sess)

        ####################################################################
        # Start to train.
        start_step = tf.train.global_step(sess, global_step)
        for step in xrange(start_step, FLAGS.max_steps):
            # Train one step.
            start_time = time.time()
            if use_data:
                _, loss_value = sess.run([train_op, loss], feed_dict={phase_train: is_train})
            else:
                image_batch, xys_batch = beziernet_data.data_generator()
                _, loss_value = sess.run([train_op, loss], feed_dict={phase_train: is_train,
                                         images: image_batch, xys: xys_batch})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Print statistics periodically.
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % 
                    (datetime.now(), step, loss_value, examples_per_sec, duration))

            # Write the summary periodically.
            if step % 100 == 0:
                if use_data:
                    summary_str, x_summary_str, y_eval = sess.run([summary_op, x_summary, logits], # logits=xys
                                                                feed_dict={phase_train: is_train})
                else:
                    summary_str, x_summary_str, y_eval = sess.run([summary_op, x_summary, logits],
                                                                feed_dict={phase_train: is_train,
                                                                images: image_batch, xys: xys_batch})
                summary_writer.add_summary(summary_str, step)
                
                new_y_img = beziernet_data.svg_to_png(y_eval, num_image=FLAGS.max_images)
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

            # Save the model checkpoint periodically.
            if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'beziernet.ckpt')
                saver.save(sess, checkpoint_path)

        print('done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('beziernet'):
        working_path = os.path.join(current_path, 'vectornet/beziernet')
        os.chdir(working_path)
        
    # create log directory    
    FLAGS.log_dir = os.path.join(FLAGS.log_dir, datetime.now().isoformat().replace(':', '-'))
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # (optional) generate bezier bin data set or extract 
    # beziernet_data.generate_bezier_bin()
    # beziernet_data.extract_bezier_bin()

    # start training
    train()

if __name__ == '__main__':
    tf.app.run()
