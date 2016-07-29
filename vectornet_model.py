# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

"""Build the VectorNet network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3,
                          """Initial learning rate.""")

# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _variable_summaries(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    with tf.name_scope('summaries'):
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        x_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        mean = tf.reduce_mean(x)
        tf.scalar_summary('mean/' + x_name, mean)
        tf.scalar_summary('stddev/' + x_name, tf.sqrt(tf.reduce_sum(tf.square(x - mean))))
        tf.scalar_summary('max/' + x_name, tf.reduce_max(x))
        tf.scalar_summary('min/' + x_name, tf.reduce_min(x))
        tf.scalar_summary('sparsity/' + x_name, tf.nn.zero_fraction(x))
        tf.histogram_summary(x_name, x)
        

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var
    

def _weight_variable(name, shape):
    """weight and summary initialization
    truncate the values more than 2 stddev and re-pick
    """
    W = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=0.1))
    _variable_summaries(W)
    return W


def _bias_variable(name, shape):
    """create a constant tensor"""
    b = _variable_on_cpu(name, shape, tf.constant_initializer(0.0))
    _variable_summaries(b)
    return b


def _batch_normalization(name, conv, d_next):
    """batch_norm/add_1, scale, offset"""
    mean, var = tf.nn.moments(conv, [0, 1, 2])
    scale = _variable_on_cpu(name + '/scale', [d_next], tf.ones_initializer) # gamma    
    offset = _variable_on_cpu(name + '/offset', [d_next], tf.zeros_initializer) # beta
    n = tf.nn.batch_normalization(conv, mean, var, offset, scale, 1e-3, name=name)
    _variable_summaries(scale)
    _variable_summaries(offset)
    _variable_summaries(n)
    return n


def _conv2d(layer_name, x, k, s, d_next):
    """down,flat-convolution"""
    with tf.variable_scope(layer_name):
        d_prev = x.get_shape()[3].value
        W = _weight_variable('weights', [k, k, d_prev, d_next])
        conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME', name='conv')
        batch = _batch_normalization('batch_norm', conv, d_next)
        return batch


def _up_conv2d(layer_name, x, k, s, d_next, out_h, out_w):
    """up-convolution"""
    with tf.variable_scope(layer_name):
        d_prev = x.get_shape()[3].value
        batch_size = tf.shape(x)[0]
        W = _weight_variable('weights', [k, k, d_prev, d_next])
        o = tf.pack([batch_size, out_h, out_w, d_next])
        conv = tf.nn.conv2d_transpose(x, W, output_shape=o, strides=[1, s, s, 1], padding='SAME', name='up_conv')
        batch = _batch_normalization('batch_norm', conv, d_next)
        return batch


def inference(x):
    """Build the Vectornet model."""
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # 1-1 down-convolutional layer: k=5x5, s=2x2, d=48
    h_conv11 = _conv2d('1-1_down', x, 5, 2, 48)
    # 1-2 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv12 = _conv2d('1-2_flat', h_conv11, 3, 1, 128)
    # 1-3 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv13 = _conv2d('1-3_flat', h_conv12, 3, 1, 128)

    # 2-1 down-convolutional layer: k=3x3, s=2x2, d=256
    h_conv21 = _conv2d('2-1_down', h_conv13, 3, 2, 256)
    # 2-2 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv22 = _conv2d('2-2_flat', h_conv21, 3, 1, 256)
    # 2-3 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv23 = _conv2d('2-3_flat', h_conv22, 3, 1, 256)

    # 3-1 down-convolutional layer: k=3x3, s=2x2, d=256
    h_conv31 = _conv2d('3-1_down', h_conv23, 3, 2, 256)
    # 3-2 flat-convolutional layer: k=3x3, s=1x1, d=512
    h_conv32 = _conv2d('3-2_flat', h_conv31, 3, 1, 512)
    # 3-3 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv33 = _conv2d('3-3_flat', h_conv32, 3, 1, 1024)
    # 3-4 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv34 = _conv2d('3-4_flat', h_conv33, 3, 1, 1024)
    # 3-5 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv35 = _conv2d('3-5_flat', h_conv34, 3, 1, 1024)
    # 3-6 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv36 = _conv2d('3-6_flat', h_conv35, 3, 1, 1024)
    # 3-7 flat-convolutional layer: k=3x3, s=1x1, d=512
    h_conv37 = _conv2d('3-7_flat', h_conv36, 3, 1, 512)
    # 3-8 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv38 = _conv2d('3-8_flat', h_conv37, 3, 1, 256)

    # 4-1 up-convolutional layer: k=4x4, s=0.5x0.5, d=256
    up_h, up_w = h_conv38.get_shape()[1].value*2, h_conv38.get_shape()[2].value*2    
    h_conv41 = _up_conv2d('4-1_up', h_conv38, 4, 2, 256, up_h, up_w)
    # 4-2 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv42 = _conv2d('4-2_flat', h_conv41, 3, 1, 256)
    # 4-3 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv43 = _conv2d('4-3_flat', h_conv42, 3, 1, 128)

    # 5-1 up-convolutional layer: k=4x4, s=0.5x0.5, d=128
    up_h, up_w = up_h*2, up_w*2
    h_conv51 = _up_conv2d('5-1_up', h_conv43, 4, 2, 128, up_h, up_w)
    # 5-2 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv52 = _conv2d('5-2_flat', h_conv51, 3, 1, 128)
    # 5-3 flat-convolutional layer: k=3x3, s=1x1, d=48
    h_conv53 = _conv2d('5-3_flat', h_conv52, 3, 1, 48)

    # 6-1 up-convolutional layer: k=4x4, s=0.5x0.5, d=48
    up_h, up_w = up_h*2, up_w*2
    h_conv61 = _up_conv2d('6-1_up', h_conv53, 4, 2, 48, up_h, up_w)
    # 6-2 flat-convolutional layer: k=3x3, s=1x1, d=24
    h_conv62 = _conv2d('6-2_flat', h_conv61, 3, 1, 24)
    # 6-3 flat-convolutional layer: k=3x3, s=1x1, d=1
    y_conv = _conv2d('6-3_flat', h_conv62, 3, 1, 1)

    return y_conv


def loss(logits, y):
    loss_sum = tf.reduce_sum(tf.square(y - logits), name='loss')
    tf.scalar_summary(loss_sum.op.name, loss_sum)
    return loss_sum


def train(total_loss, global_step):
    """Train Vectornet model.

    Create an optimizer and apply to all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
            processed.
    Returns:
        train_op: op for training.
    """
    opt = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    return apply_gradient_op