# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

"""A binary to train Vectornet on the Sketch data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time

import numpy as np
import tensorflow as tf

from temp_data import DataSet
import vectornet_model
# from vectornet_train import VectornetDataset

FLAGS = tf.app.flags.FLAGS

# parameters
tf.app.flags.DEFINE_string('log_dir', 'log/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '', #'log/2016-07-02T21-10-48.358450/vectornet.ckpt-9',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 2,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 6,
                            """Number of images to process in a batch.""")


DATA_PATH = 'data/'
RESULT_PATH = DATA_PATH + 'result2/'
LOG_PATH = 'log/'
PATCH_H, PATCH_W = 424, 424
BATCH_SIZE = 6
NUM_ITER = 600000


def train(dataset):
    """Train Vectornet for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)


        #### TEMP ####
        x = tf.placeholder(tf.float32, shape=[None, PATCH_H, PATCH_W, 1])
        h, w = PATCH_H, PATCH_W
        y = tf.placeholder(tf.float32, shape=[None, h, w, 1])
        #### TEMP ####


        # Build a Graph that computes the logits predictions from the inference model.
        logits = vectornet_model.inference(x)

        # Calculate loss.
        loss = vectornet_model.loss(logits, y)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = vectornet_model.train(loss, global_step)

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


        ### TEMP ####
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        tf.image_summary('batch_x', batch_x, max_images=2)
        tf.image_summary('batch_y', batch_y, max_images=2)
        #### TEMP ####

        # Build the summary operation.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)

        # # Start the queue runners.
        # tf.train.start_queue_runners(sess=sess)

        ####################################################################
        # Start to train.
        start_step = tf.train.global_step(sess, global_step)
        for step in xrange(start_step, FLAGS.max_steps):
            # Train one step.
            start_time = time.time()

            #### TEMP ####
            batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
            # image_summary = tf.image_summary('batch_x', batch_x, max_images=2)
            # summary_writer.add_summary(sess.run(image_summary))
            # image_summary = tf.image_summary('batch_y', batch_y, max_images=2)
            # summary_writer.add_summary(sess.run(image_summary))
            #### TEMP ####

            _, loss_value = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # Print statistics periodically.
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                print('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % (
                    datetime.now(), step, loss_value, examples_per_sec, duration))

            # Write the summary periodically.
            if step % 100 == 0:
                rec_y_t = tf.image_summary('rec_y_%d' % step, logits, max_images=6)                
                summary_str, rec_y = sess.run([summary_op, rec_y_t], feed_dict={x: batch_x, y: batch_y})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(rec_y, step)                

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.log_dir, 'vectornet.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('vectornet'):
        working_path = current_path + '/vectornet'
        os.chdir(working_path)
        
    # create log directory    
    FLAGS.log_dir += datetime.now().isoformat().replace(':', '-')
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # dataset = VectornetDataset()
    dataset = DataSet(DATA_PATH, PATCH_H, PATCH_W)
    train(dataset)

if __name__ == '__main__':
    tf.app.run()