import numpy as np
import tensorflow as tf
from ops import *
slim = tf.contrib.slim

def VDSR(x, hidden_num, repeat_num, data_format, name='VDSR',
         k=3, train=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as vs:
        for i in range(repeat_num-1):
            x = batch_norm(conv2d(x, hidden_num, data_format, k=k, s=1, act=tf.nn.relu), 
                           train, data_format, act=tf.nn.relu)
        out = batch_norm(conv2d(x, 1, data_format, k=k, s=1),
                         train, data_format, act=tf.nn.relu)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables
        
def EDSR(x, hidden_num, repeat_num, data_format, k=3, scale_factor=0.1, name='EDSR',
         train=True, reuse=False):
    ''' EDSR without upsampling '''
    with tf.variable_scope(name, reuse=reuse) as vs:
        x = conv2d(x, hidden_num, data_format, k=3, s=1)
        x_in = x

        # res. blocks
        for _ in range(repeat_num):
            x_ = x
            x = conv2d(x, hidden_num, data_format, k=3, s=1, act=tf.nn.relu)
            x = conv2d(x, hidden_num, data_format, k=3, s=1)*scale_factor
            x += x_

        x = conv2d(x, hidden_num, data_format, k=3, s=1)
        x += x_in

        out = conv2d(x, 1, data_format, k=3, s=1)
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def main(_):
    b_num = 16
    h = 64
    w = 64
    ch_num = 2

    data_format = 'NCHW'

    x = tf.placeholder(dtype=tf.float32, shape=[b_num, h, w, ch_num])
    if data_format == 'NCHW':
        x = nhwc_to_nchw(x)

    model = 2
    if model == 1:
        hidden_num = 64
        repeat_num = 20
        y = VDSR(x, hidden_num, repeat_num, data_format)
    else:
        hidden_num = 128 # 128
        repeat_num = 16 # 16
        y = EDSR(x, hidden_num, repeat_num, data_format)    
    show_all_variables()

if __name__ == '__main__':
    tf.app.run()