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
import xml.etree.ElementTree as et

import tensorflow as tf
from linenet.linenet_manager_sketch import LinenetManager


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'test/sketch',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'linenet/data/sketch',
                           """Data directory""")
tf.app.flags.DEFINE_string('file_list', 'test.txt',
                           """file_list""")
tf.app.flags.DEFINE_integer('image_width', 128, # 48-24-12-6
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 96, # 48-24-12-6
                            """Image Height.""")
tf.app.flags.DEFINE_boolean('use_batch', True,
                            """whether use batch or not""")
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """batch_size""")
tf.app.flags.DEFINE_integer('max_num_labels', 64,
                           """the maximum number of labels""")
tf.app.flags.DEFINE_integer('label_cost', 0,
                           """label cost""")
tf.app.flags.DEFINE_float('neighbor_sigma', 8,
                           """neighbor sigma""")
tf.app.flags.DEFINE_float('prediction_sigma', 0.7, # 0.7 for 0.5 threshold
                           """prediction sigma""")
tf.app.flags.DEFINE_float('window_size', 2.0,
                           """window size""")
tf.app.flags.DEFINE_boolean('compile', False,
                            """whether compile gco or not""")


def _read_svg(svg_file_path):
    with open(svg_file_path, 'r') as f:
        svg = f.read()
        svg_xml = et.fromstring(svg)
        num_paths = len(svg_xml[0]._children) - 1
        svg = svg.format(w=FLAGS.image_width, h=FLAGS.image_height)
        s_png = cairosvg.svg2png(bytestring=svg)
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity
    return s, num_paths


def _compute_accuracy(svg_file_path, labels, line_pixels):
    stroke_list = []
    with open(svg_file_path, 'r') as f:
        svg = f.read()
        svg = svg.format(w=FLAGS.image_width, h=FLAGS.image_height)

        svg_xml = et.fromstring(svg)
        num_paths = len(svg_xml[0]._children) - 1

        for i in xrange(1,num_paths+1):
            svg_xml = et.fromstring(svg)
            svg_xml[0]._children = [svg_xml[0]._children[i]]
            svg_one_stroke = et.tostring(svg_xml, method='xml')

            y_png = cairosvg.svg2png(bytestring=svg_one_stroke)
            y_img = Image.open(io.BytesIO(y_png))
            y = (np.array(y_img)[:,:,3] > 0)

            # # debug
            # y_img = np.array(y_img)[:,:,3].astype(np.float) / 255.0
            # plt.imshow(y_img, cmap=plt.cm.gray)
            # plt.show()

            stroke_list.append(y)

    acc_id_list = []
    acc_list = []
    for i in xrange(FLAGS.max_num_labels):
        label = np.nonzero(labels == i)
        # print('%d: # labels %d' % (i, len(label[0])))
        if len(label[0]) == 0:
            continue

        label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.bool)
        # for j in label:
        #     label_map[line_pixels[0][j],line_pixels[1][j]] = True
        label_map[line_pixels[0][label],line_pixels[1][label]] = True

        # # debug
        # label_map_img = label_map.astype(np.float)
        # plt.imshow(label_map_img, cmap=plt.cm.gray)
        # plt.show()

        accuracy_list = []
        for j, stroke in enumerate(stroke_list):
            intersect = np.sum(np.logical_and(label_map, stroke))
            union = np.sum(np.logical_or(label_map, stroke))
            accuracy = intersect / float(union)
            # print('compare with %d-th path, intersect: %d, union :%d, accuracy %.2f' % 
            #     (j, intersect, union, accuracy))
            accuracy_list.append(accuracy)

        id = np.argmax(accuracy_list)
        acc = np.amax(accuracy_list)
        # print('%d-th label, match to %d-th path, max: %.2f' % (i, id, acc))
        # consider only large label set
        if acc > 0.1:
            acc_id_list.append(id)
            acc_list.append(acc)

    # print('avg: %.2f' % np.average(acc_list))
    return acc_list


def graphcut(linenet_manager, file_path):
    file_name = os.path.splitext(basename(file_path))[0]
    print('%s: %s, start graphcut opt.' % (datetime.now(), file_name))

    img, num_paths = _read_svg(file_path)
    # img = _imread(file_path)

    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    start_time = time.time()
    tf.gfile.MakeDirs(FLAGS.test_dir + '/tmp')
    line_pixels = np.nonzero(img)
    num_line_pixels = len(line_pixels[0])

    dist = center = int(FLAGS.window_size * FLAGS.neighbor_sigma + 0.5)
    crop_size = int(2 * dist + 1)

    nb = NearestNeighbors(radius=dist)
    nb.fit(np.array(line_pixels).transpose())

    if FLAGS.use_batch:
        prob_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '_{id}.npy'
        linenet_manager.extract_save_crop(img, FLAGS.batch_size, prob_file_path)
        # linenet_manager.extract_save(img, FLAGS.batch_size, prob_file_path)
    else:
        y_batch, _ = linenet_manager.extract_all(img)

    duration = time.time() - start_time
    print('%s: %s, linenet process (%.3f sec)' % (datetime.now(), file_name, duration))

    # sess = tf.InteractiveSession()
    # summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.test_dir, file_name), sess.graph)

    # # original
    # img_ph = tf.placeholder(dtype=tf.float32, shape=[None, img.shape[0], img.shape[1], 1])
    # img_summary = tf.image_summary('image', img_ph, max_images=1)
    # summary_str = img_summary.eval(feed_dict={img_ph: np.reshape(1.0-img, [1, img.shape[0], img.shape[1], 1])})
    # summary_tmp = tf.Summary()
    # summary_tmp.ParseFromString(summary_str)
    # summary_tmp.value[0].tag = 'image'
    # summary_writer.add_summary(summary_tmp)

    if FLAGS.use_batch:
        map_height = map_width = crop_size
    else:
        map_height = img.shape[0]
        map_width = img.shape[1]

    # # ##################################################################################
    # # # debug: generate similarity map
    # # pred_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, map_height, map_width, 3])
    # # pred_map_summary = tf.image_summary('pred_map', pred_map_ph, max_images=1)
    # # prediction_list = []

    # if FLAGS.use_batch:
    #     for i in xrange(num_line_pixels):
    #         p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
    #         pred_p1 = np.load(prob_file_path.format(id=i))
    #         # prediction_map = np.zeros([map_height, map_width, 3], dtype=np.float)

    #         rng = nb.radius_neighbors([p1])
    #         for rj, j in enumerate(rng[1][0]): # ids
    #             if i == j:
    #                 continue
    #             p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
    #             pred_p2 = np.load(prob_file_path.format(id=j))
    #             rp2 = [center+p2[0]-p1[0],center+p2[1]-p1[1]]
    #             rp1 = [center+p1[0]-p2[0],center+p1[1]-p2[1]]
    #             pred = (pred_p1[rp2[0],rp2[1]] + pred_p2[rp1[0],rp1[1]]) * 0.5
    #             pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

    #             if i < j:
    #                 prediction_list.append(pred)

    #             d12 = rng[0][0][rj]
    #             spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
    #             pred = spatial * pred
    #             prediction_map[center+p2[0]-p1[0],center+p2[1]-p1[1]] = np.array([pred, pred, pred])

    #         prediction_map = prediction_map / np.amax(prediction_map)
    #         prediction_map[center,center] = np.array([1, 0, 0])
    #         # plt.imshow(prediction_map)
    #         # plt.show()
    #         save_path = os.path.join(FLAGS.test_dir, 'prediction_map_%d_%s' % (i, file_name))
    #         scipy.misc.imsave(save_path, prediction_map)

    # #         prediction_map = np.reshape(prediction_map, [1, map_height, map_width, 3])
    # #         summary_str = pred_map_summary.eval(feed_dict={pred_map_ph: prediction_map})
    # #         summary_tmp.ParseFromString(summary_str)
    # #         summary_tmp.value[0].tag = 'pred_map/%04d' % i
    # #         summary_writer.add_summary(summary_tmp)

    # # else:
    # #     for i in xrange(num_line_pixels):
    # #         p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
    # #         pred_p1 = np.reshape(y_batch[i,:,:,:], [map_height, map_width])
    # #         prediction_map = np.zeros([map_height, map_width, 3], dtype=np.float)

    # #         rng = nb.radius_neighbors([p1])
    # #         for rj, j in enumerate(rng[1][0]): # ids
    # #             if i == j:
    # #                 continue
    # #             p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
    # #             pred_p2 = np.reshape(y_batch[j,:,:,:], [map_height, map_width])
    # #             pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5
    # #             pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

    # #             if i < j:
    # #                 prediction_list.append(pred)

    # #             if FLAGS.neighbor_sigma > 0:
    # #                 d12 = rng[0][0][rj]
    # #                 spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
    # #                 pred = spatial * pred
    # #             prediction_map[p2[0],p2[1]] = np.array([pred, pred, pred])

    # #         prediction_map = prediction_map / np.amax(prediction_map)
    # #         prediction_map[p1[0],p1[1]] = np.array([1, 0, 0])
    # #         # plt.imshow(prediction_map)
    # #         # plt.show()
    # #         # save_path = os.path.join(FLAGS.test_dir, 'prediction_map_%d_%s' % (i, file_name))
    # #         # scipy.misc.imsave(save_path, prediction_map)

    # #         prediction_map = np.reshape(prediction_map, [1, map_height, map_width, 3])
    # #         summary_str = pred_map_summary.eval(feed_dict={pred_map_ph: prediction_map})
    # #         summary_tmp.ParseFromString(summary_str)
    # #         summary_tmp.value[0].tag = 'pred_map/%04d' % i
    # #         summary_writer.add_summary(summary_tmp)

    # # # the histogram of the data
    # # prediction_list = np.array(prediction_list)
    
    # # fig = plt.figure()
    # # weights = np.ones_like(prediction_list)/float(len(prediction_list))
    # # plt.hist(prediction_list, bins=50, color='blue', normed=False, alpha=0.75, weights=weights)
    # # plt.xlim(0, 1)
    # # plt.ylim(0, 0.5)
    # # plt.title('Histogram of Kpq')
    # # plt.grid(True)
    
    # # # Now we can save it to a numpy array.
    # # fig.canvas.draw()
    # # pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # # pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # # plt.close(fig)

    # # hist_path = os.path.join(FLAGS.test_dir, 'hist_%s_%f_%f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma))
    # # scipy.misc.imsave(hist_path, pred_hist)

    # # pred_hist = np.reshape(pred_hist, [1, pred_hist.shape[0], pred_hist.shape[1], pred_hist.shape[2]])
    # # pred_hist_ph = tf.placeholder(dtype=tf.uint8, shape=pred_hist.shape)
    # # pred_hist_summary = tf.image_summary('Kpq_hist', pred_hist_ph, max_images=1)
    
    # # summary_str = pred_hist_summary.eval(feed_dict={pred_hist_ph: pred_hist})
    # # summary_tmp.ParseFromString(summary_str)
    # # summary_tmp.value[0].tag = 'pred_Kpq_hist'
    # # summary_writer.add_summary(summary_tmp)

    # # print('Done')
    # # return
    # # ###################################################################################

    pred_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '.pred'
    f = open(pred_file_path, 'w')
    # info
    f.write(pred_file_path + '\n')
    f.write(FLAGS.data_dir + '\n')
    f.write('%d\n' % FLAGS.max_num_labels)
    f.write('%d\n' % FLAGS.label_cost)
    f.write('%f\n' % FLAGS.neighbor_sigma)
    f.write('%f\n' % FLAGS.prediction_sigma)
    f.write('%d\n' % num_line_pixels)

    # support only symmetric edge weight
    if FLAGS.use_batch:
        for i in xrange(num_line_pixels-1):
            p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
            pred_p1 = np.load(prob_file_path.format(id=i))
            rng = nb.radius_neighbors([p1])
            for rj, j in enumerate(rng[1][0]): # ids
                if j <= i:
                    continue
            # for j in xrange(i+1, num_line_pixels):
                p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
                pred_p2 = np.load(prob_file_path.format(id=j))
                rp2 = [center+p2[0]-p1[0],center+p2[1]-p1[1]]
                rp1 = [center+p1[0]-p2[0],center+p1[1]-p2[1]]
                pred = (pred_p1[rp2[0],rp2[1]] + pred_p2[rp1[0],rp1[1]]) * 0.5
                pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

                d12 = rng[0][0][rj]
                # d12 = LA.norm(p1-p2, 2)
                spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
                f.write('%d %d %f %f\n' % (i, j, pred, spatial))
    else:
        for i in xrange(num_line_pixels-1):
            p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
            pred_p1 = np.reshape(y_batch[i,:,:,:], [map_height, map_width])
            # rng = nb.radius_neighbors([p1])
            # for rj, j in enumerate(rng[1][0]): # ids
            #     if j <= i:
            #         continue
            for j in xrange(i+1, num_line_pixels):
                p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
                pred_p2 = np.reshape(y_batch[j,:,:,:], [map_height, map_width])
                pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5
                pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

                # d12 = rng[0][0][rj]
                d12 = LA.norm(p1-p2, 2)
                spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
                f.write('%d %d %f %f\n' % (i, j, pred, spatial))
    f.close()
    print('%s: %s, prediction computed' % (datetime.now(), file_name))

    # run gco_linenet
    start_time = time.time()
    working_path = os.getcwd()
    gco_path = os.path.join(working_path, 'gco/qpbo_src')
    os.chdir(gco_path)
    os.environ['LD_LIBRARY_PATH'] = os.getcwd()
    pred_fp = pred_file_path
    if pred_fp[0] != '/': # relative path
        pred_fp = '../../' + pred_fp
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
    
    # merge small label segments
    knb = NearestNeighbors(n_neighbors=7, algorithm='ball_tree')
    knb.fit(np.array(line_pixels).transpose())

    for i in xrange(FLAGS.max_num_labels):
        label = np.nonzero(labels == i)
        num_label_pixels = len(label[0])

        # if num_label_pixels > 0:
        #     print('%d: # labels %d' % (i, num_label_pixels))

        if num_label_pixels == 0 or num_label_pixels > 3:
            continue

        for l in label[0]:
            p1 = np.array([line_pixels[0][l], line_pixels[1][l]])
            _, indices = knb.kneighbors([p1], n_neighbors=7)
            max_label_nb = np.argmax(np.bincount(labels[indices][0]))
            labels[l] = max_label_nb
            # print('(%d,%d) %d -> %d' % (p1[0], p1[1], i, max_label_nb))

    # compute accuracy
    start_time = time.time()
    accuracy_list = _compute_accuracy(file_path, labels, line_pixels)
    acc_avg = np.average(accuracy_list)
    duration = time.time() - start_time
    print('%s: %s, accuracy computed, avg.: %.3f (%.3f sec)' % (datetime.now(), file_name, acc_avg, duration))

    # graphcut opt.
    u = np.unique(labels)
    num_labels = u.size
    diff_labels = num_labels - num_paths
    # print('%s: %s, label: %s' % (datetime.now(), file_name, labels))
    print('%s: %s, the number of labels %d, truth %d, diff %d' % (datetime.now(), file_name, num_labels, num_paths, diff_labels))
    print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
    print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))

    # # write summary
    # num_labels_summary = tf.scalar_summary('num_lables', tf.constant(num_labels, dtype=tf.int16))
    # summary_writer.add_summary(num_labels_summary.eval())

    # ground_truth_summary = tf.scalar_summary('ground truth', tf.constant(num_paths, dtype=tf.int16))
    # summary_writer.add_summary(ground_truth_summary.eval())

    # diff_labels_summary = tf.scalar_summary('diff', tf.constant(diff_labels, dtype=tf.int16))
    # summary_writer.add_summary(diff_labels_summary.eval())

    # # smooth_energy = tf.placeholder(dtype=tf.int32)
    # # label_energy = tf.placeholder(dtype=tf.int32)
    # # total_energy = tf.placeholder(dtype=tf.int32)
    # energy = tf.placeholder(dtype=tf.float64)
    # # smooth_energy_summary = tf.scalar_summary('smooth_energy', smooth_energy)
    # # label_energy_summary = tf.scalar_summary('label_energy', label_energy)
    # # total_energy_summary = tf.scalar_summary('total_energy', total_energy)
    # energy_summary = tf.scalar_summary('energy', energy)
    # # energy_summary = tf.merge_summary([smooth_energy_summary, label_energy_summary, total_energy_summary])
    # # # energy before optimization
    # # summary_writer.add_summary(energy_summary.eval(feed_dict={
    # #     smooth_energy:e_before[0], label_energy:e_before[1], total_energy:e_before[2]}), 0)
    # # # energy after optimization
    # # summary_writer.add_summary(energy_summary.eval(feed_dict={
    # #     smooth_energy:e_after[0], label_energy:e_after[1], total_energy:e_after[2]}), 1)
    # # energy before optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={energy:e_before}), 0)
    # # energy after optimization
    # summary_writer.add_summary(energy_summary.eval(feed_dict={energy:e_after}), 1)
    
    # duration_ph = tf.placeholder(dtype=tf.float32)
    # duration_summary = tf.scalar_summary('duration', duration_ph)
    # summary_writer.add_summary(duration_summary.eval(feed_dict={duration_ph:duration}))
    
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
    label_map_path = os.path.join(FLAGS.test_dir, 'label_map_%s_%.2f_%.2f_%d_%d_%.2f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma, num_labels, diff_labels, acc_avg))
    scipy.misc.imsave(label_map_path, label_map)
    # plt.imshow(label_map)
    # plt.show()
    
    # label_map_ph = tf.placeholder(dtype=tf.float32, shape=[None, img.shape[0], img.shape[1], 3])
    # label_map_summary = tf.image_summary('label_map', label_map_ph, max_images=1)
    # label_map = np.reshape(label_map, [1, img.shape[0], img.shape[1], 3])
    # summary_str = sess.run(label_map_summary, feed_dict={label_map_ph: label_map})
    # summary_tmp = tf.Summary()
    # summary_tmp.ParseFromString(summary_str)
    # summary_tmp.value[0].tag = 'label_map'
    # summary_writer.add_summary(summary_tmp)

    tf.gfile.DeleteRecursively(FLAGS.test_dir + '/tmp')

    return num_labels, diff_labels, acc_avg


def postprocess():
    num_files = 0
    path_list = []
    diff_list = []
    acc_list = []
    duration_list = []
    
    stat_path = os.path.join(FLAGS.data_dir, 'stat.txt')
    with open(stat_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            elif line.find('total') > -1: break

            name, num_labels, diff_labels, accuracy, duration = line.split()

    # for root, _, files in os.walk(FLAGS.test_dir):
    #     for file in files:
    #         if not file.lower().endswith('png'):
    #             continue
    #         ss = file.split('_')
    #         name = ss[2] + '_' + ss[3]
    #         num_labels = ss[6]
    #         diff_labels = ss[7]
    #         accuracy = ss[8].rstrip('.png')
    #         duration = 0

            num_labels = int(num_labels)
            diff_labels = int(diff_labels)
            accuracy = float(accuracy)
            duration = float(duration)
            num_paths = num_labels - diff_labels

            num_files = num_files + 1
            path_list.append(num_paths)
            diff_list.append(diff_labels)
            acc_list.append(accuracy)
            duration_list.append(duration)

    # the histogram of the data
    path_list = np.array(path_list)
    diff_list = np.array(diff_list)
    acc_list = np.array(acc_list)
    duration_list = np.array(duration_list)

    max_paths = np.amax(path_list)
    min_paths = np.amin(path_list)
    avg_paths = np.average(path_list)
    max_diff_labels = np.amax(diff_list)
    min_diff_labels = np.amin(diff_list)
    avg_diff_labels = np.average(np.abs(diff_list))
    max_acc = np.amax(acc_list)
    min_acc = np.amin(acc_list)
    avg_acc = np.average(acc_list)
    max_duration = np.amax(duration_list)
    min_duration = np.amin(duration_list)
    avg_duration = np.average(duration_list)
    
    bins = min(max_diff_labels - min_diff_labels, 50)
    fig = plt.figure()
    weights = np.ones_like(diff_list)/float(len(diff_list))
    plt.hist(diff_list, bins=bins, color='blue', normed=False, alpha=0.75, weights=weights)
    plt.xlim(min_diff_labels, max_diff_labels)
    plt.ylim(0, 1)
    plt.title('Histogram of Label Difference (normalized)')
    plt.grid(True)
    
    fig.canvas.draw()
    pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    hist_path = os.path.join(FLAGS.data_dir, 'label_diff_hist_norm.png')
    scipy.misc.imsave(hist_path, pred_hist)

    
    fig = plt.figure()
    plt.hist(diff_list, bins=bins, color='blue', normed=False, alpha=0.75)
    plt.xlim(min_diff_labels, max_diff_labels)
    plt.title('Histogram of Label Difference')
    plt.grid(True)
    
    fig.canvas.draw()
    pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    hist_path = os.path.join(FLAGS.data_dir, 'label_diff_hist.png')
    scipy.misc.imsave(hist_path, pred_hist)


    fig = plt.figure()
    bins = 20
    weights = np.ones_like(acc_list)/float(len(acc_list))
    plt.hist(acc_list, bins=bins, color='blue', normed=False, alpha=0.75, weights=weights)
    # plt.hist(acc_list, bins=bins, color='blue', normed=False, alpha=0.75)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Histogram of Accuracy (normalized)')
    plt.grid(True)
    
    fig.canvas.draw()
    pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    hist_path = os.path.join(FLAGS.data_dir, 'accuracy_hist_norm.png')
    scipy.misc.imsave(hist_path, pred_hist)

    
    fig = plt.figure()
    plt.hist(acc_list, bins=bins, color='blue', normed=False, alpha=0.75)
    plt.xlim(0, 1)
    plt.title('Histogram of Accuracy')
    plt.grid(True)
    
    fig.canvas.draw()
    pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    hist_path = os.path.join(FLAGS.data_dir, 'accuracy_hist.png')
    scipy.misc.imsave(hist_path, pred_hist)


    print('total # files: %d' % num_files)
    print('min/max/avg. paths: %d, %d, %.3f' % (min_paths, max_paths, avg_paths))
    print('min/max/avg. abs diff labels: %d, %d, %.3f' % (min_diff_labels, max_diff_labels, avg_diff_labels))
    print('min/max/avg. accuracy: %.3f, %.3f, %.3f' % (min_acc, max_acc, avg_acc))
    print('min/max/avg. duration (sec): %.3f, %.3f, %.3f' % (min_duration, max_duration, avg_duration))
    
    result_path = os.path.join(FLAGS.data_dir, '_result.txt')
    f = open(result_path, 'w')
    f.write('min/max/avg. paths: %d, %d, %.3f\n' % (min_paths, max_paths, avg_paths))
    f.write('min/max/avg. abs diff labels: %d, %d, %.3f\n' % (min_diff_labels, max_diff_labels, avg_diff_labels))
    f.write('min/max/avg. accuracy: %.3f, %.3f, %.3f\n' % (min_acc, max_acc, avg_acc))
    f.write('min/max/avg. duration (sec): %.3f, %.3f, %.3f\n' % (min_duration, max_duration, avg_duration))    
    f.close()


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

    if FLAGS.use_batch:
        dist = int(FLAGS.window_size * FLAGS.neighbor_sigma + 0.5)
        crop_size = 2 * dist + 1
        linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width], crop_size)
        # linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width])
    else:
        linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width])
    duration = time.time() - start_time
    print('%s: manager loaded (%.3f sec)' % (datetime.now(), duration))
    
    stat_path = os.path.join(FLAGS.test_dir, 'stat.txt')
    sf = open(stat_path, 'w')

    num_files = 0
    sum_diff_labels = 0
    min_diff_labels = 100
    max_diff_labels = -100
    diff_list = []
    sum_duration = 0
    min_duration = 10000
    max_duration = -1
    acc_avg_list = []

    if FLAGS.file_list:
        file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
        with open(file_list_path, 'r') as f:
            while True:
                line = f.readline()
                if not line: break

                file = line.rstrip()
                # file = 'n02374451_6192-3.svg_pre'
                file_path = os.path.join(FLAGS.data_dir, file)
                start_time = time.time()
                num_labels, diff_labels, acc_avg = graphcut(linenet_manager, file_path)
                sum_diff_labels = sum_diff_labels + abs(diff_labels)
                if diff_labels < min_diff_labels:
                    min_diff_labels = diff_labels
                if diff_labels > max_diff_labels:
                    max_diff_labels = diff_labels
                num_files = num_files + 1
                diff_list.append(diff_labels)            
                duration = time.time() - start_time
                sum_duration = sum_duration + duration
                if duration < min_duration:
                    min_duration = duration
                if duration > max_duration:
                    max_duration = duration
                acc_avg_list.append(acc_avg)
                print('%s:%d-%s processed (%.3f sec)' % (datetime.now(), num_files, file, duration))
                sf.write('%s %d %d %.3f %.3f\n' % (file, num_labels, diff_labels, acc_avg, duration))
    else:
        for root, _, files in os.walk(FLAGS.data_dir):
            for file in files:
                if not file.lower().endswith('svg_pre'): # 'png'):
                    continue
                
                file_path = os.path.join(root, file)
                start_time = time.time()
                num_labels, diff_labels, acc_avg = graphcut(linenet_manager, file_path)
                sum_diff_labels = sum_diff_labels + abs(diff_labels)
                if diff_labels < min_diff_labels:
                    min_diff_labels = diff_labels
                if diff_labels > max_diff_labels:
                    max_diff_labels = diff_labels
                num_files = num_files + 1
                diff_list.append(diff_labels)            
                duration = time.time() - start_time
                sum_duration = sum_duration + duration
                if duration < min_duration:
                    min_duration = duration
                if duration > max_duration:
                    max_duration = duration
                acc_avg_list.append(acc_avg)
                print('%s:%d-%s processed (%.3f sec)' % (datetime.now(), num_files, file, duration))
                sf.write('%s %d %d %.3f %.3f\n' % (file, num_labels, diff_labels, acc_avg, duration))

    # the histogram of the data
    diff_list = np.array(diff_list)
    
    # fig = plt.figure()
    # weights = np.ones_like(diff_list)/float(len(diff_list))
    # plt.hist(diff_list, bins=21, color='blue', normed=False, alpha=0.75, weights=weights)
    # plt.xlim(-5, 15)
    # plt.ylim(0, 1)
    # plt.title('Histogram of Label Difference')
    # plt.grid(True)
    
    # # Now we can save it to a numpy array.
    # fig.canvas.draw()
    # pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)

    # hist_path = os.path.join(FLAGS.test_dir, 'label_diff_hist.png')
    # scipy.misc.imsave(hist_path, pred_hist)


    # the histogram of the data
    acc_avg_list = np.array(acc_avg_list)
    
    # fig = plt.figure()
    # weights = np.ones_like(acc_avg_list)/float(len(acc_avg_list))
    # plt.hist(acc_avg_list, bins=21, color='blue', normed=False, alpha=0.75, weights=weights)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.title('Histogram of Accuracy')
    # plt.grid(True)
    
    # # Now we can save it to a numpy array.
    # fig.canvas.draw()
    # pred_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # pred_hist = pred_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)

    # hist_path = os.path.join(FLAGS.test_dir, 'acc_hist.png')
    # scipy.misc.imsave(hist_path, pred_hist)

    print('total # files: %d' % num_files)
    print('min/max/avg. abs diff labels: %d, %d, %.3f' % (min_diff_labels, max_diff_labels, sum_diff_labels/num_files))
    print('avg. accuracy: %.3f' % np.average(acc_avg_list))
    print('min/max/avg. duration (sec): %.3f, %.3f, %.3f' % (min_duration, max_duration, sum_duration/num_files))
    sf.write('total # files: %d\n' % num_files)
    sf.write('min/max/avg. abs diff labels: %d, %d, %.3f\n' % (min_diff_labels, max_diff_labels, sum_diff_labels/num_files))
    sf.write('avg. accuracy: %.3f\n' % np.average(acc_avg_list))    
    sf.write('min/max/avg. duration (sec): %.3f, %.3f, %.3f\n' % (min_duration, max_duration, sum_duration/num_files))
    sf.close()
    print('Done')


def main(_):
    # if release mode, change current path
    working_path = os.getcwd()
    if not working_path.endswith('vectornet'):
        working_path = os.path.join(working_path, 'vectornet')
        os.chdir(working_path)
    
    # make gco
    if FLAGS.compile:
        print('%s: start to compile gco' % datetime.now())
        # http://vision.csd.uwo.ca/code/
        gco_path = os.path.join(working_path, 'gco/qpbo_src')
        
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
    # postprocess()


if __name__ == '__main__':
    tf.app.run()