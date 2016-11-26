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
import io
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from numpy import linalg as LA
import scipy.stats
import scipy.misc
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import cairosvg
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
from linenet.linenet_manager_chinese import LinenetManager


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'test/chinese',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'data/chinese_128',
                           """Data directory""")
tf.app.flags.DEFINE_integer('image_width', 48,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 48,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('batch_size', 256, # 48-256, 128-32
                            """batch_size""")
tf.app.flags.DEFINE_integer('max_num_labels', 20,
                           """the maximum number of labels""")
# tf.app.flags.DEFINE_integer('label_cost', 100,
#                            """label cost""")
tf.app.flags.DEFINE_float('neighbor_sigma', 8,
                           """neighbor sigma""")
tf.app.flags.DEFINE_float('prediction_sigma', 0.7, # 0.7 for 0.5 threshold
                           """prediction sigma""")


def _imread(img_file_name, inv=False):
    """ Read, grayscale and normalize the image"""
    img = np.array(Image.open(img_file_name).convert('L')).astype(np.float) / 255.0
    if inv:
        return scipy.stats.threshold(1.0 - img, threshmax=0.0001, newval=1.0)
    else: 
        return scipy.stats.threshold(img, threshmax=0.0001, newval=1.0)


def _read_svg(svg_file_path):
    with open(svg_file_path, 'r') as f:
        svg = f.read()
        r = 0
        s = [1, -1] # c1: [1, -1], c2: [1, 1] 
        t = [0, -900] # c1: [0, -900], c2: [0, 0]
        svg = svg.format(
                w=FLAGS.image_width, h=FLAGS.image_height,
                r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        s_png = cairosvg.svg2png(bytestring=svg)
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) / 255.0
    return s


def graphcut(linenet_manager, file_path):
    file_name = os.path.splitext(basename(file_path))[0]
    print('%s: %s, start graphcut opt.' % (datetime.now(), file_name))
    
    img = _read_svg(file_path)
    # img = _imread(file_path)

    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()
    
    start_time = time.time()
    y_batch, line_pixels = linenet_manager.extract_all(img)
    num_line_pixels = len(line_pixels[0])
    dist = center = 2 * FLAGS.neighbor_sigma
    nb = NearestNeighbors(radius=dist)
    nb.fit(np.array(line_pixels).transpose())
    tf.gfile.MakeDirs(FLAGS.test_dir + '/tmp')
    duration = time.time() - start_time
    print('%s: %s, linenet process (%.3f sec)' % (datetime.now(), file_name, duration))
    
    sess = tf.InteractiveSession()
    summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.test_dir, file_name), sess.graph)

    # original
    img_ph = tf.placeholder(dtype=tf.float32, shape=[None, img.shape[0], img.shape[1], 1])
    img_summary = tf.image_summary('image', img_ph, max_images=1)
    summary_str = img_summary.eval(feed_dict={img_ph: np.reshape(1.0-img, [1, img.shape[0], img.shape[1], 1])})
    summary_tmp = tf.Summary()
    summary_tmp.ParseFromString(summary_str)
    summary_tmp.value[0].tag = 'image'
    summary_writer.add_summary(summary_tmp)

    ###################################################################################
    # # debug: generate similarity map
    # pred_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, crop_size, crop_size, 3])
    # pred_map_summary = tf.image_summary('pred_map', pred_map_ph, max_images=1)
    # prediction_list = []

    # for i in xrange(num_line_pixels):
    #     p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
    #     pred_p1 = np.reshape(y_batch[i,:,:,:], [img.shape[0], img.shape[1]])
    #     prediction_map = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.float)
        
    #     for j in xrange(num_line_pixels):
    #         if i == j:
    #             continue
    #         p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
    #         pred_p2 = np.reshape(y_batch[j,:,:,:], [img.shape[0], img.shape[1]])
    #         pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5                        
    #         pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

    #         if i < j:
    #             prediction_list.append(pred)

    #         if FLAGS.neighbor_sigma > 0:
    #             d12 = LA.norm(p1-p2, 2)
    #             spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
    #             pred = spatial * pred
    #         prediction_map[p2[0],p2[1]] = np.array([pred, pred, pred])

    #     prediction_map = prediction_map / np.amax(prediction_map)
    #     prediction_map[p1[0],p1[1]] = np.array([1, 0, 0])
    #     # plt.imshow(prediction_map)
    #     # plt.show()
    #     # save_path = os.path.join(FLAGS.test_dir, 'prediction_map_%d_%s' % (i, file_name))
    #     # scipy.misc.imsave(save_path, prediction_map)

    #     prediction_map = np.reshape(prediction_map, [1, crop_size, crop_size, 3])
    #     summary_str = pred_map_summary.eval(feed_dict={pred_map_ph: prediction_map})
    #     summary_tmp.ParseFromString(summary_str)
    #     summary_tmp.value[0].tag = 'pred_map/%04d' % i
    #     summary_writer.add_summary(summary_tmp)

    # # the histogram of the data
    # prediction_list = np.array(prediction_list)
    
    # fig = plt.figure()
    # weights = np.ones_like(prediction_list)/float(len(prediction_list))
    # plt.hist(prediction_list, bins=50, color='blue', normed=False, alpha=0.75, weights=weights)
    # plt.xlim(0, 1)
    # plt.ylim(0, 0.5)
    # plt.title('Histogram of Kpq')
    # plt.grid(True)
    
    # # Now we can save it to a numpy array.
    # fig.canvas.draw()
    # pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)

    # hist_path = os.path.join(FLAGS.test_dir, 'hist_%s_%f_%f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma))
    # scipy.misc.imsave(hist_path, pred_hist)

    # pred_hist = np.reshape(pred_hist, [1, pred_hist.shape[0], pred_hist.shape[1], pred_hist.shape[2]])
    # pred_hist_ph = tf.placeholder(dtype=tf.uint8, shape=pred_hist.shape)
    # pred_hist_summary = tf.image_summary('Kpq_hist', pred_hist_ph, max_images=1)
    
    # summary_str = pred_hist_summary.eval(feed_dict={pred_hist_ph: pred_hist})
    # summary_tmp.ParseFromString(summary_str)
    # summary_tmp.value[0].tag = 'pred_Kpq_hist'
    # summary_writer.add_summary(summary_tmp)

    # print('Done')
    # return
    # ###################################################################################

    pred_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '.pred'
    f = open(pred_file_path, 'w')
    # info
    f.write(pred_file_path + '\n')
    f.write(FLAGS.data_dir + '\n')
    f.write('%d\n' % FLAGS.max_num_labels)
    # f.write('%d\n' % FLAGS.label_cost)
    f.write('%f\n' % FLAGS.neighbor_sigma)
    f.write('%f\n' % FLAGS.prediction_sigma)
    f.write('%d\n' % num_line_pixels)
    
    # support only symmetric edge weight
    for i in xrange(num_line_pixels-1):
        p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
        pred_p1 = np.reshape(y_batch[i,:,:,:], [img.shape[0], img.shape[1]])
        for j in xrange(i+1, num_line_pixels):
            p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
            pred_p2 = np.reshape(y_batch[j,:,:,:], [img.shape[0], img.shape[1]])
            pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5
            pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)
            
            if FLAGS.neighbor_sigma > 0:
                d12 = LA.norm(p1-p2, 2)
                spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
            else:
                spatial = 1.0
            f.write('%d %d %f %f\n' % (i, j, pred, spatial))

    f.close()
    print('%s: %s, prediction computed' % (datetime.now(), file_name))
    
    # run gco_linenet
    start_time = time.time()
    working_path = os.getcwd()
    gco_path = os.path.join(working_path, 'gco/gco_src')
    os.chdir(gco_path)
    os.environ['LD_LIBRARY_PATH'] = os.getcwd()
    pred_fp = pred_file_path
    if pred_file_path[0] != '/': # relative path
        pred_fp = '../../'
    call(['./gco_linenet', pred_fp])
    os.chdir(working_path)
    
    # read result
    label_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '.label'
    f = open(label_file_path, 'r')
    e_before = float(f.readline())
    e_after = float(f.readline())
    labels = np.fromstring(f.read(), dtype=np.int32, sep=' ')
    f.close()
    # os.remove(pred_file_path)
    # os.remove(label_file_path)
    duration = time.time() - start_time
    print('%s: %s, labeling finished (%.3f sec)' % (datetime.now(), file_name, duration))
    
    
    # graphcut opt.
    u = np.unique(labels)
    num_labels = u.size
    # print('%s: %s, label: %s' % (datetime.now(), file_name, labels))
    print('%s: %s, the number of labels %d' % (datetime.now(), file_name, num_labels))
    print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
    print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))
    
    # write summary
    num_labels_summary = tf.scalar_summary('num_lables', tf.constant(num_labels, dtype=tf.int16))
    summary_writer.add_summary(num_labels_summary.eval())

    # smooth_energy = tf.placeholder(dtype=tf.int32)
    # label_energy = tf.placeholder(dtype=tf.int32)
    # total_energy = tf.placeholder(dtype=tf.int32)
    energy = tf.placeholder(dtype=tf.float64)
    # smooth_energy_summary = tf.scalar_summary('smooth_energy', smooth_energy)
    # label_energy_summary = tf.scalar_summary('label_energy', label_energy)
    # total_energy_summary = tf.scalar_summary('total_energy', total_energy)
    energy_summary = tf.scalar_summary('energy', energy)
    # energy_summary = tf.merge_summary([smooth_energy_summary, label_energy_summary, total_energy_summary])
    # # energy before optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={
    #     smooth_energy:e_before[0], label_energy:e_before[1], total_energy:e_before[2]}), 0)
    # # energy after optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={
    #     smooth_energy:e_after[0], label_energy:e_after[1], total_energy:e_after[2]}), 1)
    # energy before optimization
    summary_writer.add_summary(energy_summary.eval(feed_dict={energy:e_before}), 0)
    # energy after optimization
    summary_writer.add_summary(energy_summary.eval(feed_dict={energy:e_after}), 1)
    
    duration_ph = tf.placeholder(dtype=tf.float32)
    duration_summary = tf.scalar_summary('duration', duration_ph)
    summary_writer.add_summary(duration_summary.eval(feed_dict={duration_ph:duration}))
    
    # save label map image
    cmap = plt.get_cmap('jet')    
    cnorm  = colors.Normalize(vmin=0, vmax=num_labels-1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    label_map = np.ones([img.shape[0], img.shape[1], 3], dtype=np.float)
    for i in xrange(num_line_pixels):
        # color = cscalarmap.to_rgba(labels[i])
        color = cscalarmap.to_rgba(np.where(u==labels[i])[0])[0]

        # print(line_pixels[0][i],line_pixels[1][i],labels[i]) # ,color)
        label_map[line_pixels[0][i],line_pixels[1][i]] = color[:3]
    
    # debug
    label_map_path = os.path.join(FLAGS.test_dir, 'label_map_%s_%.2f_%.2f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma))
    scipy.misc.imsave(label_map_path, label_map)
    # plt.imshow(label_map)
    # plt.show()
    
    label_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, img.shape[0], img.shape[1], 3])
    label_map_summary = tf.image_summary('label_map', label_map_ph, max_images=1)
    label_map = np.reshape(label_map, [1, img.shape[0], img.shape[1], 3])
    summary_str = sess.run(label_map_summary, feed_dict={label_map_ph: label_map})
    summary_tmp = tf.Summary()
    summary_tmp.ParseFromString(summary_str)
    summary_tmp.value[0].tag = 'label_map'
    summary_writer.add_summary(summary_tmp)

    tf.gfile.DeleteRecursively(FLAGS.test_dir + '/tmp')

    return num_labels


def graphcut_(linenet_manager, file_path):
    file_name = os.path.splitext(basename(file_path))[0]
    print('%s: %s, start graphcut opt.' % (datetime.now(), file_name))
    
    img = _read_svg(file_path)
    # img = _imread(file_path)

    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()
    
    start_time = time.time()
    line_pixels = np.nonzero(img)
    num_line_pixels = len(line_pixels[0])
    dist = center = 2 * FLAGS.neighbor_sigma
    crop_size = 2 * dist + 1
    nb = NearestNeighbors(radius=dist)
    nb.fit(np.array(line_pixels).transpose())
    
    tf.gfile.MakeDirs(FLAGS.test_dir + '/tmp')
    prob_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '_{id}.npy'
    linenet_manager.extract_save_crop(img, FLAGS.batch_size, prob_file_path)
    duration = time.time() - start_time
    print('%s: %s, linenet process (%.3f sec)' % (datetime.now(), file_name, duration))
    
    sess = tf.InteractiveSession()
    summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.test_dir, file_name), sess.graph)

    # original
    img_ph = tf.placeholder(dtype=tf.float32, shape=[None, img.shape[0], img.shape[1], 1])
    img_summary = tf.image_summary('image', img_ph, max_images=1)
    summary_str = img_summary.eval(feed_dict={img_ph: np.reshape(1.0-img, [1, img.shape[0], img.shape[1], 1])})
    summary_tmp = tf.Summary()
    summary_tmp.ParseFromString(summary_str)
    summary_tmp.value[0].tag = 'image'
    summary_writer.add_summary(summary_tmp)

    # ###################################################################################
    # # debug: generate similarity map
    # pred_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, crop_size, crop_size, 3])
    # pred_map_summary = tf.image_summary('pred_map', pred_map_ph, max_images=1)
    # prediction_list = []

    # for i in xrange(num_line_pixels):
    #     p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
    #     pred_p1 = np.load(prob_file_path.format(id=i))
    #     prediction_map = np.zeros([crop_size, crop_size, 3], dtype=np.float)

    #     rng = nb.radius_neighbors([p1])
    #     for rj, j in enumerate(rng[1][0]): # ids
    #         if i == j:
    #             continue
    #         p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
    #         pred_p2 = np.load(prob_file_path.format(id=j))
    #         rp2 = [center+p2[0]-p1[0],center+p2[1]-p1[1]]
    #         rp1 = [center+p1[0]-p2[0],center+p1[1]-p2[1]]
    #         pred = (pred_p1[rp2[0],rp2[1]] + pred_p2[rp1[0],rp1[1]]) * 0.5
    #         pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

    #         if i < j:
    #             prediction_list.append(pred)

    #         d12 = rng[0][0][rj]
    #         spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
    #         pred = spatial * pred
    #         prediction_map[center+p2[0]-p1[0],center+p2[1]-p1[1]] = np.array([pred, pred, pred])

    #     prediction_map = prediction_map / np.amax(prediction_map)
    #     prediction_map[center,center] = np.array([1, 0, 0])
    #     # plt.imshow(prediction_map)
    #     # plt.show()
    #     # save_path = os.path.join(FLAGS.test_dir, 'prediction_map_%d_%s' % (i, file_name))
    #     # scipy.misc.imsave(save_path, prediction_map)

    #     prediction_map = np.reshape(prediction_map, [1, crop_size, crop_size, 3])
    #     summary_str = pred_map_summary.eval(feed_dict={pred_map_ph: prediction_map})
    #     summary_tmp.ParseFromString(summary_str)
    #     summary_tmp.value[0].tag = 'pred_map/%04d' % i
    #     summary_writer.add_summary(summary_tmp)

    # # the histogram of the data
    # prediction_list = np.array(prediction_list)
    
    # fig = plt.figure()
    # weights = np.ones_like(prediction_list)/float(len(prediction_list))
    # plt.hist(prediction_list, bins=50, color='blue', normed=False, alpha=0.75, weights=weights)
    # plt.xlim(0, 1)
    # plt.ylim(0, 0.5)
    # plt.title('Histogram of Kpq')
    # plt.grid(True)
    
    # # Now we can save it to a numpy array.
    # fig.canvas.draw()
    # pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)

    # hist_path = os.path.join(FLAGS.test_dir, 'hist_%s_%f_%f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma))
    # scipy.misc.imsave(hist_path, pred_hist)

    # pred_hist = np.reshape(pred_hist, [1, pred_hist.shape[0], pred_hist.shape[1], pred_hist.shape[2]])
    # pred_hist_ph = tf.placeholder(dtype=tf.uint8, shape=pred_hist.shape)
    # pred_hist_summary = tf.image_summary('Kpq_hist', pred_hist_ph, max_images=1)
    
    # summary_str = pred_hist_summary.eval(feed_dict={pred_hist_ph: pred_hist})
    # summary_tmp.ParseFromString(summary_str)
    # summary_tmp.value[0].tag = 'pred_Kpq_hist'
    # summary_writer.add_summary(summary_tmp)

    # print('Done')
    # return
    # ###################################################################################

    pred_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '.pred'
    f = open(pred_file_path, 'w')
    # info
    f.write(pred_file_path + '\n')
    f.write(FLAGS.data_dir + '\n')
    f.write('%d\n' % FLAGS.max_num_labels)
    # f.write('%d\n' % FLAGS.label_cost)
    f.write('%f\n' % FLAGS.neighbor_sigma)
    f.write('%f\n' % FLAGS.prediction_sigma)
    f.write('%d\n' % num_line_pixels)
    
    # support only symmetric edge weight
    for i in xrange(num_line_pixels-1):
        p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
        pred_p1 = np.load(prob_file_path.format(id=i))
        rng = nb.radius_neighbors([p1])
        for rj, j in enumerate(rng[1][0]): # ids
            if i >= j:
                continue
            p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
            pred_p2 = np.load(prob_file_path.format(id=j))
            rp2 = [center+p2[0]-p1[0],center+p2[1]-p1[1]]
            rp1 = [center+p1[0]-p2[0],center+p1[1]-p2[1]]
            pred = (pred_p1[rp2[0],rp2[1]] + pred_p2[rp1[0],rp1[1]]) * 0.5
            pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

            d12 = rng[0][0][rj]
            spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
            f.write('%d %d %f %f\n' % (i, j, pred, spatial))

    f.close()
    print('%s: %s, prediction computed' % (datetime.now(), file_name))
    
    # run gco_linenet
    start_time = time.time()
    working_path = os.getcwd()
    gco_path = os.path.join(working_path, 'gco/gco_src')
    os.chdir(gco_path)
    os.environ['LD_LIBRARY_PATH'] = os.getcwd()
    pred_fp = pred_file_path
    if pred_file_path[0] != '/': # relative path
        pred_fp = '../../'
    call(['./gco_linenet', pred_fp])
    os.chdir(working_path)
    
    # read result
    label_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '.label'
    f = open(label_file_path, 'r')
    e_before = float(f.readline())
    e_after = float(f.readline())
    labels = np.fromstring(f.read(), dtype=np.int32, sep=' ')
    f.close()
    # os.remove(pred_file_path)
    # os.remove(label_file_path)
    duration = time.time() - start_time
    print('%s: %s, labeling finished (%.3f sec)' % (datetime.now(), file_name, duration))
    
    
    # graphcut opt.
    u = np.unique(labels)
    num_labels = u.size
    # print('%s: %s, label: %s' % (datetime.now(), file_name, labels))
    print('%s: %s, the number of labels %d' % (datetime.now(), file_name, num_labels))
    print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
    print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))
    
    # write summary
    num_labels_summary = tf.scalar_summary('num_lables', tf.constant(num_labels, dtype=tf.int16))
    summary_writer.add_summary(num_labels_summary.eval())

    # smooth_energy = tf.placeholder(dtype=tf.int32)
    # label_energy = tf.placeholder(dtype=tf.int32)
    # total_energy = tf.placeholder(dtype=tf.int32)
    energy = tf.placeholder(dtype=tf.float64)
    # smooth_energy_summary = tf.scalar_summary('smooth_energy', smooth_energy)
    # label_energy_summary = tf.scalar_summary('label_energy', label_energy)
    # total_energy_summary = tf.scalar_summary('total_energy', total_energy)
    energy_summary = tf.scalar_summary('energy', energy)
    # energy_summary = tf.merge_summary([smooth_energy_summary, label_energy_summary, total_energy_summary])
    # # energy before optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={
    #     smooth_energy:e_before[0], label_energy:e_before[1], total_energy:e_before[2]}), 0)
    # # energy after optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={
    #     smooth_energy:e_after[0], label_energy:e_after[1], total_energy:e_after[2]}), 1)
    # energy before optimization
    summary_writer.add_summary(energy_summary.eval(feed_dict={energy:e_before}), 0)
    # energy after optimization
    summary_writer.add_summary(energy_summary.eval(feed_dict={energy:e_after}), 1)
    
    duration_ph = tf.placeholder(dtype=tf.float32)
    duration_summary = tf.scalar_summary('duration', duration_ph)
    summary_writer.add_summary(duration_summary.eval(feed_dict={duration_ph:duration}))
    
    # save label map image
    cmap = plt.get_cmap('jet')    
    cnorm  = colors.Normalize(vmin=0, vmax=num_labels-1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    label_map = np.ones([img.shape[0], img.shape[1], 3], dtype=np.float)
    for i in xrange(num_line_pixels):
        # color = cscalarmap.to_rgba(labels[i])
        color = cscalarmap.to_rgba(np.where(u==labels[i])[0])[0]

        # print(line_pixels[0][i],line_pixels[1][i],labels[i]) # ,color)
        label_map[line_pixels[0][i],line_pixels[1][i]] = color[:3]
    
    # debug
    label_map_path = os.path.join(FLAGS.test_dir, 'label_map_%s_%.2f_%.2f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma))
    scipy.misc.imsave(label_map_path, label_map)
    # plt.imshow(label_map)
    # plt.show()
    
    label_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, img.shape[0], img.shape[1], 3])
    label_map_summary = tf.image_summary('label_map', label_map_ph, max_images=1)
    label_map = np.reshape(label_map, [1, img.shape[0], img.shape[1], 3])
    summary_str = sess.run(label_map_summary, feed_dict={label_map_ph: label_map})
    summary_tmp = tf.Summary()
    summary_tmp.ParseFromString(summary_str)
    summary_tmp.value[0].tag = 'label_map'
    summary_writer.add_summary(summary_tmp)

    tf.gfile.DeleteRecursively(FLAGS.test_dir + '/tmp')

    return num_labels


def parameter_tune():
    # create managers
    start_time = time.time()
    print('%s: manager loading...' % datetime.now())
    linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width])
    duration = time.time() - start_time
    print('%s: manager loaded (%.3f sec)' % (datetime.now(), duration))
    
    f = open('label_parameter_0.2_0.9_8_0.1_1.0_10.txt', 'w')
            
    for root, _, files in os.walk(FLAGS.data_dir):
        for file in files:
            if not file.lower().endswith('svg'):
                continue

            min_np = [0,0]
            min_labels = 100
            # n_sig_list = [1]
            # p_sig_list = [0.1]
            n_sig_list = np.linspace(0.2, 0.9, 8).tolist()
            # n_sig_list = [8]
            p_sig_list = np.linspace(0.1, 1.0, 10).tolist()
            # p_sig_list = [0.7] #, 0.764, 0.765]
            for n_sig in n_sig_list:
                FLAGS.neighbor_sigma = n_sig
                print('n_sig: %.4f' % FLAGS.neighbor_sigma)
                for p_sig in p_sig_list:
                    FLAGS.prediction_sigma = p_sig
                    print('p_sig: %.4f' % FLAGS.prediction_sigma)
                    file_path = os.path.join(root, file)
                    start_time = time.time()
                    num_labels = graphcut(linenet_manager, file_path)
                    duration = time.time() - start_time
                    print('%s: %s processed (%.3f sec)' % (datetime.now(), file, duration))

                    f.write('%d [%d, %0.4f]\n' % (num_labels, n_sig, p_sig))

                    if min_labels > num_labels:
                        min_labels = num_labels
                        min_np = [n_sig, p_sig]
                        print('!!!!min', min_labels, min_np)

                    # if num_labels < 20 and num_labels > 10:
                    #     print('Find!')
                    #     return

            print('!!!!min', min_labels, min_np)
            f.write('min %d [%d, %0.4f]' % (min_labels, min_np[0], min_np[1]))
            
    f.close()
    print('Done')


def test():
    # create managers
    start_time = time.time()
    print('%s: manager loading...' % datetime.now())
    linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width])
    # dist = 2 * FLAGS.neighbor_sigma
    # crop_size = 2 * dist + 1
    # linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width], crop_size)    
    duration = time.time() - start_time
    print('%s: manager loaded (%.3f sec)' % (datetime.now(), duration))
    
    for root, _, files in os.walk(FLAGS.data_dir):
        for file in files:
            if not file.lower().endswith('svg_pre'): # 'png'):
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
    
    # make gco
    print('%s: start to compile gco' % datetime.now())
    # http://vision.csd.uwo.ca/code/
    gco_path = os.path.join(working_path, 'gco/gco_src')
    
    os.chdir(gco_path)
    call(['make', 'rm'])
    call(['make'])
    call(['make', 'gco_linenet'])
    os.chdir(working_path)
    print('%s: gco compiled' % datetime.now())

    # create test directory
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)
    
    # start
    test()
    # parameter_tune()


if __name__ == '__main__':
    tf.app.run()