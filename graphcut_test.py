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
from os.path import basename
import time
from subprocess import call

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from numpy import linalg as LA
import scipy.stats
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import cairosvg

import tensorflow as tf
from linenet.linenet_manager import LinenetManager
from gco_python import pygco


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'test/graphcut',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'data/graphcut', 
                           """Data directory""")
tf.app.flags.DEFINE_integer('prediction_scale', 100,
                           """make it sensible value when rounding""")
tf.app.flags.DEFINE_integer('max_num_labels', 10, 
                           """the maximum number of labels""")


def _imread(img_file_name, inv=False):
    """ Read, grayscale and normalize the image"""
    img = np.array(Image.open(img_file_name).convert('L')).astype(np.float) / 255.0
    if inv: 
        return 1.0 - img
    else: 
        return img

def graphcut(linenet_manager, file_path):
    print('%s: start graphcut opt.' % datetime.now())
    img = _imread(file_path, inv=True)

    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # compute probability map of all line pixels
    y_batch, line_pixels = linenet_manager.extract_all(img)
    
    # specify neighbor weights
    num_line_pixels = len(line_pixels[0])
    file_name = os.path.splitext(basename(file_path))[0]

    sess = tf.InteractiveSession()
    summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.test_dir, file_name), sess.graph)
    pred_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3])
    pred_map_summary = tf.image_summary('pred_map', pred_map_ph, max_images=1)

    ###################################################################################
    # debug: generate similarity map
    for i in xrange(num_line_pixels):
        p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
        pred_p1 = np.reshape(y_batch[i,:,:,:], [FLAGS.image_size, FLAGS.image_size])
        prediction_map = np.zeros([FLAGS.image_size, FLAGS.image_size, 3], dtype=np.float)
        
        for j in xrange(num_line_pixels):
            if i == j:
                continue
            p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
            pred_p2 = np.reshape(y_batch[j,:,:,:], [FLAGS.image_size, FLAGS.image_size])
            avg_pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5            
            prediction_map[p2[0],p2[1]] = np.array([avg_pred, avg_pred, avg_pred])

        prediction_map[p1[0],p1[1]] = np.array([1, 0, 0])
        # plt.imshow(prediction_map)
        # plt.show()
        # save_path = os.path.join(FLAGS.test_dir, 'prediction_map_%d_%s' % (i, file_name))
        # scipy.misc.imsave(save_path, prediction_map)

        prediction_map = np.reshape(prediction_map, [1, FLAGS.image_size, FLAGS.image_size, 3])
        summary_str = pred_map_summary.eval(feed_dict={pred_map_ph: prediction_map})
        summary_tmp = tf.Summary()
        summary_tmp.ParseFromString(summary_str)        
        summary_tmp.value[0].tag = 'pred_map/%04d' % i
        summary_writer.add_summary(summary_tmp)

    # print('Done')
    # return
    # ###################################################################################

    # support only symmetric edge weight
    edge_weight = []
    for i in xrange(num_line_pixels-1):
        p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
        pred_p1 = np.reshape(y_batch[i,:,:,:], [FLAGS.image_size, FLAGS.image_size])
        for j in xrange(i+1, num_line_pixels):
            p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
            pred_p2 = np.reshape(y_batch[j,:,:,:], [FLAGS.image_size, FLAGS.image_size])
            avg_pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5
            scaled_avg_pred = avg_pred * 2.0 * FLAGS.prediction_scale - FLAGS.prediction_scale
            edge_weight.append([i, j, scaled_avg_pred])    
    edge_weight = np.array(edge_weight).astype(np.int32)

    # graphcut opt.
    data_term = np.zeros([num_line_pixels*num_line_pixels, FLAGS.max_num_labels], dtype=np.int32)
    pairwise = np.ones([FLAGS.max_num_labels, FLAGS.max_num_labels], dtype=np.int32) - 2 * np.eye(FLAGS.max_num_labels, dtype=np.int32)
    label_cost = 10000
    result_label = pygco.cut_from_graph(edge_weight, data_term, pairwise)
    num_labels = np.amax(result_label) + 1
    print('%s: %s, the number of labels %d' % (datetime.now(), file_name, num_labels))
    # print('%s: %s, energy before optimization, smooth %d, label %d, total %d' % (datetime.now(), file_name,  
    #     e_before[0], e_before[1], e_before[2]))
    # print('%s: %s, energy after optimization, data %d, smooth %d, label %d, total %d' % (datetime.now(), file_name, 
    #     e_after[0], e_after[1], e_after[2]))
    
    # write summary
    num_labels_summary = tf.scalar_summary('num_lables', tf.constant(num_labels, dtype=tf.int16))
    summary_writer.add_summary(num_labels_summary.eval())

    # smooth_energy = tf.placeholder(dtype=tf.int32)
    # label_energy = tf.placeholder(dtype=tf.int32)
    # total_energy = tf.placeholder(dtype=tf.int32)
    # smooth_energy_summary = tf.scalar_summary('smooth_energy', smooth_energy)
    # label_energy_summary = tf.scalar_summary('label_energy', label_energy)
    # total_energy_summary = tf.scalar_summary('total_energy', total_energy)
    # energy_summary = tf.merge_summary([smooth_energy_summary, label_energy_summary, total_energy_summary])
    # # energy before optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={
    #     smooth_energy:e_before[0], label_energy:e_before[1], total_energy:e_before[2]}), 0)
    # # energy after optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={
    #     smooth_energy:e_after[0], label_energy:e_after[1], total_energy:e_after[2]}), 1)
    
    
    # save label map image
    cmap = plt.get_cmap('jet')
    cnorm  = colors.Normalize(vmin=0, vmax=num_labels-1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    label_map = np.ones([FLAGS.image_size, FLAGS.image_size, 3], dtype=np.float)
    for i in xrange(num_line_pixels):
        color = cscalarmap.to_rgba(result_label[i])
        label_map[line_pixels[0][i],line_pixels[1][i]] = color[:3]
    
    # label_map_path = os.path.join(FLAGS.test_dir, 'label_map_%s.png' % file_name)
    # scipy.misc.imsave(label_map_path, label_map)
    label_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3])
    label_map_summary = tf.image_summary('label_map', label_map_ph, max_images=1)
    label_map = np.reshape(label_map, [1, FLAGS.image_size, FLAGS.image_size, 3])
    summary_str = sess.run(label_map_summary, feed_dict={label_map_ph: label_map})
    summary_tmp = tf.Summary()
    summary_tmp.ParseFromString(summary_str)        
    summary_tmp.value[0].tag = 'label_map'
    summary_writer.add_summary(summary_tmp)


def test():
    # create managers
    start_time = time.time()
    linenet_manager = LinenetManager()
    duration = time.time() - start_time
    print('%s: Linenet manager loaded (%.3f sec)' % (datetime.now(), duration))
    
    for root, _, files in os.walk(FLAGS.data_dir):
        for file in files:
            if not file.lower().endswith('png'):
                continue
            
            file_path = os.path.join(root, file)
            start_time = time.time()
            graphcut(linenet_manager, file_path)
            duration = time.time() - start_time
            print('%s: %s processed (%.3f sec)' % (datetime.now(), file, duration))

    print('Done')


def main(_):
    # if release mode, change current path
    working_path = os.getcwd()
    if not working_path.endswith('vectornet'):
        working_path = os.path.join(working_path, 'vectornet')
        os.chdir(working_path)
    
    # make pygco
    print('%s: start compile pygco' % datetime.now())
    pygco_path = os.path.join(working_path, 'gco_python')
    os.chdir(pygco_path)
    call(['make'])
    os.chdir(working_path)    
    print('%s: pygco compiled' % datetime.now())    

    # create test directory
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)

    # start
    test()


if __name__ == '__main__':
    tf.app.run()