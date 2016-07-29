import os
import time

import numpy as np
import scipy.misc

import tensorflow as tf

from data_set import DataSet


DATA_PATH = 'data/'
RESULT_PATH = DATA_PATH + 'result2/'
LOG_PATH = 'log/'
PATCH_H, PATCH_W = 424, 424
BATCH_SIZE = 6
NUM_ITER = 600000


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

# weight initialization    
def weight_variable(shape):
    # truncate the values more than 2 stddev and re-pick
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # create a constant tensor
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_normalization(conv, d_next):
    mean, var = tf.nn.moments(conv, [0, 1, 2])
    scale = tf.Variable(tf.ones([d_next])) # gamma
    offset = tf.Variable(tf.zeros([d_next])) # beta    
    return tf.nn.batch_normalization(conv, mean, var, offset, scale, 1e-3)

# down,flat-convolution
def conv2d(layer_name, x, k, s, d_next):
    with tf.name_scope(layer_name):
        d_prev = x.get_shape()[3].value
        with tf.name_scope('weights'):
            W = weight_variable([k, k, d_prev, d_next])
            # variable_summaries(W, layer_name + '/weights')
        # b = bias_variable([d_next])
        # return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME') + b
        
        with tf.name_scope('conv'):
            conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
            # tf.histogram_summary(layer_name + '/conv', conv)            
        with tf.name_scope('batch'):
            batch = batch_normalization(conv, d_next)
            # tf.histogram_summary(layer_name + '/batch', batch)
        return batch

# up-convolution
def up_conv2d(layer_name, x, k, s, d_next, out_h, out_w):
    with tf.name_scope(layer_name):
        d_prev = x.get_shape()[3].value
        batch_size = tf.shape(x)[0]
        with tf.name_scope('weights'):
            W = weight_variable([k, k, d_next, d_prev])
            # variable_summaries(W, layer_name + '/weights')
        o = tf.pack([batch_size, out_h, out_w, d_next])
        # b = bias_variable([d_next])
        # return tf.nn.conv2d_transpose(x, W, output_shape=o, strides=[1, s, s, 1], padding='SAME') + b
        
        with tf.name_scope('conv'):
            conv = tf.nn.conv2d_transpose(x, W, output_shape=o, strides=[1, s, s, 1], padding='SAME')
            # tf.histogram_summary(layer_name + '/conv', conv)
        with tf.name_scope('batch'):
            batch = batch_normalization(conv, d_next)
            # tf.histogram_summary(layer_name + '/batch', batch)
        return batch


def main():
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('vectornet'):
        working_path = current_path + '/vectornet'
        os.chdir(working_path)


    np.random.seed(0)
    input_data = DataSet(DATA_PATH, PATCH_H, PATCH_W)

    sess = tf.InteractiveSession()

    #######################################################################
    x = tf.placeholder(tf.float32, shape=[None, PATCH_H, PATCH_W, 1])
    
    # 1-1 down-convolutional layer: k=5x5, s=2x2, d=48
    h_conv11 = conv2d('1-1_down', x, 5, 2, 48)
    
    
    # h_conv12 = conv2d('1-2_flat', h_conv11, 3, 1, 24)
    # h_conv13 = conv2d('1-3_flat', h_conv12, 3, 1, 12)
    # h_conv14 = conv2d('1-4_down', h_conv13, 3, 2, 12)
    # h_conv14_h, h_conv14_w = h_conv14.get_shape()[1].value, h_conv14.get_shape()[2].value  
    # h_conv15 = up_conv2d('1-5_up', h_conv14, 4, 2, 12, h_conv14_h*2, h_conv14_w*2)
    # h_conv16 = conv2d('1-6_flat', h_conv15, 3, 1, 6)
    # h_conv17 = up_conv2d('1-7_up', h_conv16, 4, 2, 6, h_conv14_h*4, h_conv14_w*4)
    # y_conv = conv2d('1-8_flat', h_conv17, 3, 1, 1)


    # 1-2 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv12 = conv2d('1-2_flat', h_conv11, 3, 1, 128)
    # 1-3 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv13 = conv2d('1-3_flat', h_conv12, 3, 1, 128)

    # 2-1 down-convolutional layer: k=3x3, s=2x2, d=256
    h_conv21 = conv2d('2-1_down', h_conv13, 3, 2, 256)
    # 2-2 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv22 = conv2d('2-2_flat', h_conv21, 3, 1, 256)
    # 2-3 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv23 = conv2d('2-3_flat', h_conv22, 3, 1, 256)

    # 3-1 down-convolutional layer: k=3x3, s=2x2, d=256
    h_conv31 = conv2d('3-1_down', h_conv23, 3, 2, 256)
    # 3-2 flat-convolutional layer: k=3x3, s=1x1, d=512
    h_conv32 = conv2d('3-2_flat', h_conv31, 3, 1, 512)
    # 3-3 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv33 = conv2d('3-3_flat', h_conv32, 3, 1, 1024)
    # 3-4 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv34 = conv2d('3-4_flat', h_conv33, 3, 1, 1024)
    # 3-5 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv35 = conv2d('3-5_flat', h_conv34, 3, 1, 1024)
    # 3-6 flat-convolutional layer: k=3x3, s=1x1, d=1024
    h_conv36 = conv2d('3-6_flat', h_conv35, 3, 1, 1024)
    # 3-7 flat-convolutional layer: k=3x3, s=1x1, d=512
    h_conv37 = conv2d('3-7_flat', h_conv36, 3, 1, 512)
    # 3-8 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv38 = conv2d('3-8_flat', h_conv37, 3, 1, 256)

    # 4-1 up-convolutional layer: k=4x4, s=0.5x0.5, d=256
    up_h, up_w = h_conv38.get_shape()[1].value*2, h_conv38.get_shape()[2].value*2    
    h_conv41 = up_conv2d('4-1_up', h_conv38, 4, 2, 256, up_h, up_w)
    # 4-2 flat-convolutional layer: k=3x3, s=1x1, d=256
    h_conv42 = conv2d('4-2_flat', h_conv41, 3, 1, 256)
    # 4-3 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv43 = conv2d('4-3_flat', h_conv42, 3, 1, 128)

    # 5-1 up-convolutional layer: k=4x4, s=0.5x0.5, d=128
    up_h, up_w = up_h*2, up_w*2
    h_conv51 = up_conv2d('5-1_up', h_conv43, 4, 2, 128, up_h, up_w)
    # 5-2 flat-convolutional layer: k=3x3, s=1x1, d=128
    h_conv52 = conv2d('5-2_flat', h_conv51, 3, 1, 128)
    # 5-3 flat-convolutional layer: k=3x3, s=1x1, d=48
    h_conv53 = conv2d('5-3_flat', h_conv52, 3, 1, 48)

    # 6-1 up-convolutional layer: k=4x4, s=0.5x0.5, d=48
    up_h, up_w = up_h*2, up_w*2
    h_conv61 = up_conv2d('6-1_up', h_conv53, 4, 2, 48, up_h, up_w)
    # 6-2 flat-convolutional layer: k=3x3, s=1x1, d=24
    h_conv62 = conv2d('6-2_flat', h_conv61, 3, 1, 24)
    # 6-3 flat-convolutional layer: k=3x3, s=1x1, d=1
    y_conv = conv2d('6-3_flat', h_conv62, 3, 1, 1)

    
    h, w = PATCH_H, PATCH_W
    y_ = tf.placeholder(tf.float32, shape=[None, h, w, 1])    

    #######################################################################

    loss = tf.reduce_sum(tf.square(y_ - y_conv))
    # tf.scalar_summary('loss', loss)
    # optimizer = tf.train.AdadeltaOptimizer(1e-4).minimize(loss)
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    accuracy = tf.reduce_sum(tf.square(y_ - y_conv))
    saver = tf.train.Saver()

    # # Merge all the summaries and write them out
    # merged = tf.merge_all_summaries()
    # train_writer = tf.train.SummaryWriter(LOG_PATH + '/train', sess.graph)
    # # test_writer = tf.train.SummaryWriter(LOG_PATH + '/test')


    print "initialize"
    tf.initialize_all_variables().run()

    print "start training"
    start = time.time()
    for step in range(NUM_ITER):
        if step % 100 == 0 or step == NUM_ITER - 1:
            # test
            # # save cropped patches
            # batch_x, batch_y = input_data.next_batch(BATCH_SIZE)
            # bx = np.reshape(batch_x, [-1, PATCH_H, PATCH_W])
            # by = np.reshape(batch_y, [-1, PATCH_H, PATCH_W])
            # for j in range(BATCH_SIZE):            
            #     scipy.misc.imsave(RESULT_PATH+"x-%d-%d.png" % (step, j), bx[j, ...])
            #     scipy.misc.imsave(RESULT_PATH+"y-%d-%d.png" % (step, j), by[j, ...]) 
            
            batch_x, batch_y = input_data.test_x, input_data.test_y
            
            # ### !!!!!! TEMP for downscaling
            # bb = np.reshape(batch_y, [-1, PATCH_H, PATCH_W])
            # bbb = np.empty([BATCH_SIZE, h, w])
            # for j in range(BATCH_SIZE):            
            #     bbb[j, ...] = scipy.misc.imresize(bb[j, ...], (h, w)) / 255.0
            #     # scipy.misc.imsave(RESULT_PATH+"y-%d-%d.png" % (i, j), bbb[j, ...]) 
            # batch_y = np.reshape(bbb, [-1, h, w, 1])
            # ### !!!!!! TEMP

            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            
            result_imgs = y_conv.eval(feed_dict={x: batch_x, y_: batch_y})
            result_imgs = np.reshape(result_imgs, [-1, h, w])
            for j in range(BATCH_SIZE):
                result = np.clip(result_imgs[j, ...], 0.0, 1.0)
                # result = np.clip(result_imgs[j, ...], 0, 255).astype(np.uint8)
                # scipy.misc.imshow(result)
                scipy.misc.imsave(RESULT_PATH + "result-%d-%d.png" % (step, j), result)

            end = time.time()
            elapsed = end - start
            start = time.time()      
            m, s = divmod(elapsed, 60)
            hr, m = divmod(m, 60)
            print "Step %d, Training Accuracy %g, Elapsed Time: %d:%d:%d [hms]" % (step, train_accuracy, hr, m, s)

            if step == NUM_ITER - 1:
                save_path = saver.save(sess, RESULT_PATH + "vars-%d-%d-%d.ckpt" % (hr, m, s))
                print "Model saved in file: %s" % save_path

        else:
            # train
            batch_x, batch_y = input_data.next_batch(BATCH_SIZE)

            # ### !!!!!! TEMP for downscaling
            # bb = np.reshape(batch_y, [-1, PATCH_H, PATCH_W])
            # bbb = np.empty([BATCH_SIZE, h, w])
            # for j in range(BATCH_SIZE):            
            #     bbb[j, ...] = scipy.misc.imresize(bb[j, ...], (h, w)) / 255.0
            #     # scipy.misc.imsave(RESULT_PATH+"y-%d-%d.png" % (i, j), bbb[j, ...]) 
            # batch_y = np.reshape(bbb, [-1, h, w, 1])
            # ### !!!!!! TEMP        
            
            optimizer.run(feed_dict={x: batch_x, y_: batch_y})
            # summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_x, y_: batch_y})
            # train_writer.add_summary(summary, step)

    # close the session
    sess.close()


if __name__ == '__main__':
    main()