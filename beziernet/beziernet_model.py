# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

"""Build the BezierNet network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import beziernet_data


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
        x_name = x.op.name # re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
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


def _batch_normalization(name, x, d_next, phase_train, is_conv=True):
    """batch_norm/add_1, scale, offset"""
    if is_conv:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    scale = _variable_on_cpu(name + '/scale', [d_next], tf.ones_initializer) # gamma    
    offset = _variable_on_cpu(name + '/offset', [d_next], tf.zeros_initializer) # beta
    
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def _mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        _mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    
    n = tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-3, name=name)
    _variable_summaries(scale)
    _variable_summaries(offset)
    _variable_summaries(n)
    return n


def _conv2d(layer_name, x, k, s, d_next, phase_train):
    """down, flat-convolution layer"""
    with tf.variable_scope(layer_name):
        d_prev = x.get_shape()[3].value
        W = _weight_variable('1_weights', [k, k, d_prev, d_next])
        conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME', name='1_conv')
        _variable_summaries(conv)
        batch = _batch_normalization('2_batch_norm', conv, d_next, phase_train)
        relu = tf.nn.relu(batch, name='3_relu')
        _variable_summaries(relu)
        return relu

def _fc(layer_name, x, d_next, phase_train, use_activation=True):
    """fully-connected layer"""
    with tf.variable_scope(layer_name):
        num_v_prev = x.get_shape()[1].value
        W = _weight_variable('1_weights', [num_v_prev, d_next])
        fc = tf.matmul(x, W, name='2_matmul')
        _variable_summaries(fc)
        batch = _batch_normalization('3_batch_norm', fc, d_next, phase_train, is_conv=False)
        if use_activation:
            relu = tf.nn.relu(batch, name='4_relu')
            _variable_summaries(relu)
            return relu
        else:
            return batch


def inference(images, phase_train):
    """Build the Bezier model."""
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #   
    # 1-1 down-convolutional layer: k=3x3, s=2x2, d=64, 96 -> 48
    h_conv11 = _conv2d('1-1_down', images, 3, 2, 64, phase_train)
    # 1-2 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv12 = _conv2d('1-2_flat', h_conv11, 3, 1, 128, phase_train)

    # 2-1 down-convolutional layer: k=3x3, s=2x2, d=128 -> 24
    h_conv21 = _conv2d('2-1_down', h_conv12, 3, 2, 128, phase_train)
    # 2-2 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv22 = _conv2d('2-2_flat', h_conv21, 3, 1, 256, phase_train)

    # 3-1 down-convolutional layer: k=3x3, s=2x2, d=256 -> 12
    h_conv31 = _conv2d('3-1_down', h_conv22, 3, 2, 256, phase_train)
    # 3-2 flat-convolutional layer: k=3x3, s=1x1, d=512
    h_conv32 = _conv2d('3-2_flat', h_conv31, 3, 1, 512, phase_train)

    # 4-1 down-convolutional layer: k=3x3, s=2x2, d=512 -> 6
    h_conv41 = _conv2d('4-1_down', h_conv32, 3, 2, 512, phase_train)
    # 4-2 flat-convolutional layer: k=3x3, s=1x1, d=512
    h_conv42 = _conv2d('4-2_flat', h_conv41, 3, 1, 512, phase_train)
    
    h_conv42_dim = h_conv42.get_shape()[1].value*h_conv42.get_shape()[2].value*h_conv42.get_shape()[3].value
    h_conv42_flat = tf.reshape(h_conv42, [-1, h_conv42_dim])
        
    # 5-1 fully-connected layer: d=1024
    h_fc51 = _fc('5-1_fc', h_conv42_flat, 1024, phase_train)
    # 5-2 fully-connected layer: d=512
    h_fc52 = _fc('5-2_fc', h_fc51, 512, phase_train)
    # 5-3 fully-connected layer: d=256
    h_fc53 = _fc('5-3_fc', h_fc52, 256, phase_train)
    # 5-4 fully-connected layer: d=8
    y_fc = _fc('5-4_fc', h_fc53, 8, phase_train, use_activation=False)
    
    return y_fc


def loss(logits, xys):
    loss_mean = tf.reduce_mean(tf.square(xys - logits), name='loss')
    return loss_mean