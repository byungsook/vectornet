# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
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
    """batch_norm/1_scale, 2_offset, 3_batch"""
    if is_conv:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    scale = _variable_on_cpu(name+'/1_scale', [d_next], tf.ones_initializer) # gamma
    offset = _variable_on_cpu(name+'/2_offset', [d_next], tf.zeros_initializer) # beta
    
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def _mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train,
                        _mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    
    n = tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-3, name=name+'/3_batch')
    _variable_summaries(scale)
    _variable_summaries(offset)
    _variable_summaries(n)
    return n


def _conv2d(layer_name, x, k, s, d_next, phase_train):
    """down, flat-convolution layer"""
    with tf.variable_scope(layer_name):
        d_prev = x.get_shape()[3].value
        W = _weight_variable('1_filter_weights', [k, k, d_prev, d_next])
        conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME', name='2_conv_feature')
        _variable_summaries(conv)
        batch = _batch_normalization('3_batch_norm', conv, d_next, phase_train)
        relu = tf.nn.relu(batch, name='4_relu')
        _variable_summaries(relu)
        return relu


def _res_unit(layer_name, x, k, s, d_next, phase_train, use_elu=True, use_res=True):
    with tf.variable_scope(layer_name):
        d_prev = x.get_shape()[3].value
        W = _weight_variable('1_filter_weights', [k, k, d_prev, d_next])
        conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME', name='2_conv_feature')
        _variable_summaries(conv)
        if use_elu:
            elu = tf.nn.elu(conv, name='3_elu')
            _variable_summaries(elu)
            return elu
        else:
            batch = _batch_normalization('3_batch_norm', conv, d_next, phase_train)
            if use_res:
                return x + batch
            else:
                return batch


def model3(x, x_no_p, phase_train):
    y_hat = tf.mul(model1(x, x_no_p, phase_train), x_no_p, name='21_mul')
    _variable_summaries(y_hat)
    return y_hat


def model2(x, x_no_p, phase_train):
    layer01 = _res_unit('01_res', x,       3, 1, 64, phase_train)
    layer02 = _res_unit('02_res', layer01, 3, 1, 64, phase_train, use_elu=False)
    layer03 = _res_unit('03_res', layer02, 3, 1, 64, phase_train)
    layer04 = _res_unit('04_res', layer03, 3, 1, 64, phase_train, use_elu=False)
    layer05 = _res_unit('05_res', layer04, 3, 1, 64, phase_train)
    layer06 = _res_unit('06_res', layer05, 3, 1, 64, phase_train, use_elu=False)
    layer07 = _res_unit('07_res', layer06, 3, 1, 64, phase_train)
    layer08 = _res_unit('08_res', layer07, 3, 1, 64, phase_train, use_elu=False)
    layer09 = _res_unit('09_res', layer08, 3, 1, 64, phase_train)
    layer10 = _res_unit('10_res', layer09, 3, 1, 64, phase_train, use_elu=False)
    layer11 = _res_unit('11_res', layer10, 3, 1, 64, phase_train)
    layer12 = _res_unit('12_res', layer11, 3, 1, 64, phase_train, use_elu=False)
    layer13 = _res_unit('13_res', layer12, 3, 1, 64, phase_train)
    layer14 = _res_unit('14_res', layer13, 3, 1, 64, phase_train, use_elu=False)
    layer15 = _res_unit('15_res', layer14, 3, 1, 64, phase_train)
    layer16 = _res_unit('16_res', layer15, 3, 1, 64, phase_train, use_elu=False)
    layer17 = _res_unit('17_res', layer16, 3, 1, 64, phase_train)
    layer18 = _res_unit('18_res', layer17, 3, 1, 64, phase_train, use_elu=False)
    layer19 = _res_unit('19_res', layer18, 3, 1, 64, phase_train)
    layer20 = _res_unit('20_res', layer19, 3, 1,  1, phase_train, use_elu=False, use_res=False)
    #layer21 = tf.nn.sigmoid(layer20, name='21_sig')
    layer21 = tf.nn.relu(layer20, name='21_relu')
    _variable_summaries(layer21)
    y_hat = tf.mul(layer21, x_no_p, name='22_mul')
    _variable_summaries(y_hat)
    return y_hat


def model1(x, x_no_p, phase_train):
    # all flat-convolutional layer: k=3x3, s=1x1, d=64
    h_conv01 = _conv2d('01_flat', x,        3, 1, 64, phase_train)
    h_conv02 = _conv2d('02_flat', h_conv01, 3, 1, 64, phase_train)
    h_conv03 = _conv2d('03_flat', h_conv02, 3, 1, 64, phase_train)
    h_conv04 = _conv2d('04_flat', h_conv03, 3, 1, 64, phase_train)
    h_conv05 = _conv2d('05_flat', h_conv04, 3, 1, 64, phase_train)
    h_conv06 = _conv2d('06_flat', h_conv05, 3, 1, 64, phase_train)
    h_conv07 = _conv2d('07_flat', h_conv06, 3, 1, 64, phase_train)
    h_conv08 = _conv2d('08_flat', h_conv07, 3, 1, 64, phase_train)
    h_conv09 = _conv2d('09_flat', h_conv08, 3, 1, 64, phase_train)
    h_conv10 = _conv2d('10_flat', h_conv09, 3, 1, 64, phase_train)
    h_conv11 = _conv2d('11_flat', h_conv10, 3, 1, 64, phase_train)
    h_conv12 = _conv2d('12_flat', h_conv11, 3, 1, 64, phase_train)
    h_conv13 = _conv2d('13_flat', h_conv12, 3, 1, 64, phase_train)
    h_conv14 = _conv2d('14_flat', h_conv13, 3, 1, 64, phase_train)
    h_conv15 = _conv2d('15_flat', h_conv14, 3, 1, 64, phase_train)
    h_conv16 = _conv2d('16_flat', h_conv15, 3, 1, 64, phase_train)
    h_conv17 = _conv2d('17_flat', h_conv16, 3, 1, 64, phase_train)
    h_conv18 = _conv2d('18_flat', h_conv17, 3, 1, 64, phase_train)
    h_conv19 = _conv2d('19_flat', h_conv18, 3, 1, 64, phase_train)
    h_conv20 = _conv2d('20_flat', h_conv19, 3, 1,  1, phase_train)    
    return h_conv20


def inference(x, x_no_p, phase_train, model=1):
    model_selector = {
        1: model1,
        2: model2,
        3: model3,
    }
    return model_selector[model](x, x_no_p, phase_train)


def loss(y_hat, y):
    # y_hat: estimate, y: training set
    l2_loss = tf.nn.l2_loss(y_hat - y, name='l2_loss')
    return l2_loss
