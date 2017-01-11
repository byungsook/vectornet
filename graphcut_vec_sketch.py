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
from skimage import measure

import tensorflow as tf
from linenet.linenet_manager_sketch import LinenetManager
from linenet.linenet_manager_intersect import IntersectnetManager


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'test/sketch',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'linenet/data/sketch',
                           """Data directory""")
tf.app.flags.DEFINE_string('file_list', '',
                           """file_list""")
tf.app.flags.DEFINE_integer('num_test_files', 1,
                           """num_test_files""")
tf.app.flags.DEFINE_integer('image_width', 64, # 64-96-128
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 48, # 48-72-96
                            """Image Height.""")
tf.app.flags.DEFINE_boolean('use_batch', True,
                            """whether use batch or not""")
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """batch_size""")
tf.app.flags.DEFINE_integer('max_num_labels', 100,
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
tf.app.flags.DEFINE_boolean('use_intersect', True,
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


def _compute_accuracy(svg_file_path, labels, line_pixels, num_line_pixels, dup_rev_dict):
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
        i_label_list = np.nonzero(labels == i)
        num_i_label_pixels = len(i_label_list[0])

        if num_i_label_pixels == 0:
            continue

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_line_pixels:
                i_label_list[0][j] = dup_rev_dict[i_label]

        i_label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.bool)
        i_label_map[line_pixels[0][i_label_list],line_pixels[1][i_label_list]] = True

        # # debug
        # i_label_map_img = i_label_map.astype(np.float)
        # plt.imshow(i_label_map_img, cmap=plt.cm.gray)
        # plt.show()

        accuracy_list = []
        for j, stroke in enumerate(stroke_list):
            intersect = np.sum(np.logical_and(i_label_map, stroke))
            union = np.sum(np.logical_or(i_label_map, stroke))
            accuracy = intersect / float(union)
            # print('compare with %d-th path, intersect: %d, union :%d, accuracy %.2f' % 
            #     (j, intersect, union, accuracy))
            accuracy_list.append(accuracy)

        id = np.argmax(accuracy_list)
        acc = np.amax(accuracy_list)
        # print('%d-th label, match to %d-th path, max: %.2f' % (i, id, acc))
        # consider only large label set
        # if acc > 0.1:
        acc_id_list.append(id)
        acc_list.append(acc)

    # print('avg: %.2f' % np.average(acc_list))
    return acc_list


def graphcut(linenet_manager, intersectnet_manager, file_path):
    file_name = os.path.splitext(basename(file_path))[0]
    print('%s: %s, start graphcut opt.' % (datetime.now(), file_name))

    # convert svg to raster image
    img, num_paths = _read_svg(file_path)

    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()


    # predict using linenet
    start_time = time.time()
    
    tf.gfile.MakeDirs(FLAGS.test_dir + '/tmp')
    if FLAGS.use_batch:
        prob_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '_{id}.npy'
        # linenet_manager.extract_save_crop(img, FLAGS.batch_size, prob_file_path)
        linenet_manager.extract_save(img, FLAGS.batch_size, prob_file_path)
    else:
        y_batch, _ = linenet_manager.extract_all(img)

    duration = time.time() - start_time
    print('%s: %s, linenet process (%.3f sec)' % (datetime.now(), file_name, duration))


    # for neighbor search
    dist = center = int(FLAGS.window_size * FLAGS.neighbor_sigma + 0.5)
    crop_size = int(2 * dist + 1)
    line_pixels = np.nonzero(img)
    num_line_pixels = len(line_pixels[0])
    nb = NearestNeighbors(radius=dist)
    nb.fit(np.array(line_pixels).transpose())
  
    if FLAGS.use_batch:
        map_height = map_width = crop_size
    else:
        map_height = img.shape[0]
        map_width = img.shape[1]

    
    # predict intersection using intersection net
    start_time = time.time()
    intersect = (intersectnet_manager.intersect(img) > 0)
    intersect = np.reshape(intersect[0,:,:,:], [FLAGS.image_height, FLAGS.image_width])

    intersect_map_path = os.path.join(FLAGS.test_dir, 'intersect_map_%s.png' % file_name)
    scipy.misc.imsave(intersect_map_path, intersect)

    # # debug
    # plt.imshow(intersect, cmap=plt.cm.gray)
    # plt.show()

    dup_dict = {}
    dup_rev_dict = {}
    dup_id = num_line_pixels # start id of duplicated pixels
    if FLAGS.use_intersect:
        for i in xrange(num_line_pixels):
            p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
            if intersect[line_pixels[0][i], line_pixels[1][i]]:
                dup_dict[i] = dup_id
                dup_rev_dict[dup_id] = i
                dup_id += 1

    # # debug
    # print(dup_dict)
    # print(dup_rev_dict)

    duration = time.time() - start_time
    print('%s: %s, intersectnet process (%.3f sec)' % (datetime.now(), file_name, duration))

    
    # write config file for graphcut
    start_time = time.time()
    pred_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '.pred'
    f = open(pred_file_path, 'w')
    # info
    f.write(pred_file_path + '\n')
    f.write(FLAGS.data_dir + '\n')
    f.write('%d\n' % FLAGS.max_num_labels)
    f.write('%d\n' % FLAGS.label_cost)
    f.write('%f\n' % FLAGS.neighbor_sigma)
    f.write('%f\n' % FLAGS.prediction_sigma)
    # f.write('%d\n' % num_line_pixels)
    f.write('%d\n' % dup_id)

    # support only symmetric edge weight
    if FLAGS.use_batch:
        for i in xrange(num_line_pixels-1):
            p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
            pred_p1 = np.load(prob_file_path.format(id=i))
            # rng = nb.radius_neighbors([p1])
            # neighbor_list = rng[1][0]
            # for rj, j in enumerate(neighbor_list): # ids
            #     if j <= i:
            #         continue
            for j in xrange(i+1, num_line_pixels): # see entire neighbors
                p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
                pred_p2 = np.load(prob_file_path.format(id=j))
                # rp2 = [center+p2[0]-p1[0],center+p2[1]-p1[1]]
                # rp1 = [center+p1[0]-p2[0],center+p1[1]-p2[1]]
                # pred = (pred_p1[rp2[0],rp2[1]] + pred_p2[rp1[0],rp1[1]]) * 0.5
                pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5 # see entire neighbors
                pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

                # d12 = rng[0][0][rj]
                d12 = LA.norm(p1-p2, 2) # see entire neighbors
                spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
                f.write('%d %d %f %f\n' % (i, j, pred, spatial))

                dup_i = dup_dict.get(i)
                if dup_i is not None:
                    f.write('%d %d %f %f\n' % (j, dup_i, pred, spatial)) # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (i, dup_i, -1000, 1)) # might need to set negative pred rather than 0
                dup_j = dup_dict.get(j)
                if dup_j is not None:
                    f.write('%d %d %f %f\n' % (i, dup_j, pred, spatial)) # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (j, dup_j, -1000, 1)) # might need to set negative pred rather than 0

                if dup_i is not None and dup_j is not None:
                    f.write('%d %d %f %f\n' % (dup_i, dup_j, pred, spatial)) # dup_i < dup_j
            
            # outside_list = np.setxor1d(xrange(num_line_pixels), neighbor_list)
            # for j in outside_list:
            #     if j <= i: continue
            #     else: f.write('%d %d %f %f\n' % (i, j, 0, 0))
    else:
        for i in xrange(num_line_pixels-1):
            p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
            pred_p1 = np.reshape(y_batch[i,:,:,:], [map_height, map_width])
            # rng = nb.radius_neighbors([p1])
            # neighbor_list = rng[1][0]
            # for rj, j in enumerate(neighbor_list): # ids
            #     if j <= i:
            #         continue
            for j in xrange(i+1, num_line_pixels): # see entire neighbors
                p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
                pred_p2 = np.reshape(y_batch[j,:,:,:], [map_height, map_width])
                pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5
                pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

                # d12 = rng[0][0][rj]
                d12 = LA.norm(p1-p2, 2) # see entire neighbors
                spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
                f.write('%d %d %f %f\n' % (i, j, pred, spatial))

                dup_i = dup_dict.get(i)
                if dup_i is not None:
                    f.write('%d %d %f %f\n' % (j, dup_i, pred, spatial)) # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (i, dup_i, -1000, 1)) # might need to set negative pred rather than 0
                dup_j = dup_dict.get(j)
                if dup_j is not None:
                    f.write('%d %d %f %f\n' % (i, dup_j, pred, spatial)) # as dup is always smaller than normal id
                    f.write('%d %d %f %f\n' % (j, dup_j, -1000, 1)) # might need to set negative pred rather than 0

                if dup_i is not None and dup_j is not None:
                    f.write('%d %d %f %f\n' % (dup_i, dup_j, pred, spatial)) # dup_i < dup_j

            # outside_list = np.setxor1d(xrange(num_line_pixels), neighbor_list)
            # for j in outside_list:
            #     if j <= i: continue
            #     else: f.write('%d %d %f %f\n' % (i, j, 0, 0))

    f.close()
    duration = time.time() - start_time
    print('%s: %s, prediction computed (%.3f sec)' % (datetime.now(), file_name, duration))

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

    # read graphcut result
    label_file_path = os.path.join(FLAGS.test_dir + '/tmp', file_name) + '.label'
    f = open(label_file_path, 'r')
    e_before = float(f.readline())
    e_after = float(f.readline())
    labels = np.fromstring(f.read(), dtype=np.int32, sep=' ')
    f.close()
    duration = time.time() - start_time
    print('%s: %s, labeling finished (%.3f sec)' % (datetime.now(), file_name, duration))


    # merge small label segments
    knb = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knb.fit(np.array(line_pixels).transpose())

    for iter in xrange(2):
        # # debug
        # print('%d-th iter' % iter)
        for i in xrange(FLAGS.max_num_labels):
            i_label_list = np.nonzero(labels == i)
            num_i_label_pixels = len(i_label_list[0])

            if num_i_label_pixels == 0:
                continue

            # handle duplicated pixels
            for j, i_label in enumerate(i_label_list[0]):
                if i_label >= num_line_pixels:
                    i_label_list[0][j] = dup_rev_dict[i_label]

            # connected component analysis on 'i' label map
            i_label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.float)
            i_label_map[line_pixels[0][i_label_list],line_pixels[1][i_label_list]] = 1.0
            cc_map, num_cc = measure.label(i_label_map, background=0, return_num=True)

            # # debug
            # print('%d: # labels %d, # cc %d' % (i, num_i_label_pixels, num_cc))
            # plt.imshow(cc_map, cmap='spectral')
            # plt.show()

            # detect small pixel component
            for j in xrange(num_cc):
                j_cc_list = np.nonzero(cc_map == (j+1))
                num_j_cc = len(j_cc_list[0])

                # consider only less than 5 pixels component
                if num_j_cc > 4:
                    continue

                # assign dominant label of neighbors using knn
                for k in xrange(num_j_cc):
                    p1 = np.array([j_cc_list[0][k], j_cc_list[1][k]])
                    _, indices = knb.kneighbors([p1], n_neighbors=5)
                    max_label_nb = np.argmax(np.bincount(labels[indices][0]))
                    labels[indices[0][0]] = max_label_nb

                    # # debug
                    # print(' (%d,%d) %d -> %d' % (p1[0], p1[1], i, max_label_nb))

                    dup = dup_dict.get(indices[0][0])
                    if dup is not None:
                        labels[dup] = max_label_nb

    # print result
    u = np.unique(labels)
    num_labels = u.size
    diff_labels = num_labels - num_paths
    accuracy_list = _compute_accuracy(file_path, labels, line_pixels, num_line_pixels, dup_rev_dict)
    acc_avg = np.average(accuracy_list)
    
    print('%s: %s, the number of labels %d, truth %d, diff %d' % (datetime.now(), file_name, num_labels, num_paths, diff_labels))
    print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
    print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))
    print('%s: %s, accuracy computed, avg.: %.3f' % (datetime.now(), file_name, acc_avg))


    # save label map image
    cmap = plt.get_cmap('jet')    
    cnorm  = colors.Normalize(vmin=0, vmax=num_labels-1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    
    label_map = np.ones([FLAGS.image_height, FLAGS.image_width, 3], dtype=np.float)
    first_svg = True
    target_svg_path = os.path.join(FLAGS.test_dir, 'label_map_svg_%s_%d_%d_%.2f.svg' % (file_name, num_labels, diff_labels, acc_avg))
    for i in xrange(FLAGS.max_num_labels):
        i_label_list = np.nonzero(labels == i)
        num_label_pixels = len(i_label_list[0])

        if num_label_pixels == 0:
            continue

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_line_pixels:
                i_label_list[0][j] = dup_rev_dict[i_label]

        color = cscalarmap.to_rgba(np.where(u==i)[0])[0]
        label_map[line_pixels[0][i_label_list],line_pixels[1][i_label_list]] = color[:3]

        # save i label map
        i_label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.int)
        i_label_map[line_pixels[0][i_label_list],line_pixels[1][i_label_list]] = 1
        _, num_cc = measure.label(i_label_map, background=0, return_num=True)
        i_label_map_path = os.path.join(FLAGS.test_dir + '/tmp', 'i_label_map_%s_%d_%d.bmp' % (file_name, i, num_cc))
        scipy.misc.imsave(i_label_map_path, i_label_map)

        # vectorize using potrace
        color *= 255
        color_hex = '#%02x%02x%02x' % (color[0], color[1], color[2])
        call(['potrace', '-s', '-i', '-C'+color_hex, i_label_map_path])
        
        # # morphology transform
        # if num_cc > 1:
        #     i_label_map = scipy.ndimage.morphology.binary_closing(i_label_map, 
        #         structure=np.ones((7,7)), iterations=1)

        #     i_label_map_before = os.path.join(FLAGS.test_dir + '/tmp', 'i_label_map_%s_%d_%d.bmp' % (file_name, i, num_cc))
        #     i_label_map_new = os.path.join(FLAGS.test_dir + '/tmp', 'i_label_map_%s_%d_%d_before.bmp' % (file_name, i, num_cc))
        #     call(['cp', i_label_map_before, i_label_map_new])
        #     i_label_svg_before = os.path.join(FLAGS.test_dir + '/tmp', 'i_label_map_%s_%d_%d.svg' % (file_name, i, num_cc))
        #     i_label_svg_new = os.path.join(FLAGS.test_dir + '/tmp', 'i_label_map_%s_%d_%d_before.svg' % (file_name, i, num_cc))
        #     call(['cp', i_label_svg_before, i_label_svg_new])

        #     scipy.misc.imsave(i_label_map_path, i_label_map)
        #     call(['potrace', '-s', '-i', '-C'+color_hex, i_label_map_path])

        i_label_map_svg = os.path.join(FLAGS.test_dir + '/tmp', 'i_label_map_%s_%d_%d.svg' % (file_name, i, num_cc))
        if first_svg:
            call(['cp', i_label_map_svg, target_svg_path])
            first_svg = False
        else:
            with open(target_svg_path, 'r') as f:
                target_svg = f.read()

            with open(i_label_map_svg, 'r') as f:
                source_svg = f.read()

            path_start = source_svg.find('<g')
            path_end = source_svg.find('</svg>')

            insert_pos = target_svg.find('</svg>')            
            target_svg = target_svg[:insert_pos] + source_svg[path_start:path_end] + target_svg[insert_pos:]

            with open(target_svg_path, 'w') as f:
                f.write(target_svg)

    label_map_path = os.path.join(FLAGS.test_dir, 'label_map_%s_%.2f_%.2f_%d_%d_%.2f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma, num_labels, diff_labels, acc_avg))
    scipy.misc.imsave(label_map_path, label_map)

    # # debug
    # plt.imshow(label_map)
    # plt.show()
      
    # tf.gfile.DeleteRecursively(FLAGS.test_dir + '/tmp')
    return num_labels, diff_labels, acc_avg


def postprocess(stat_dir):
    num_files = 0
    path_list = []
    diff_list = []
    acc_list = []
    duration_list = []
    
    stat_path = os.path.join(stat_dir, 'stat.txt')
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

    hist_path = os.path.join(stat_dir, 'label_diff_hist_norm.png')
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

    hist_path = os.path.join(stat_dir, 'label_diff_hist.png')
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

    hist_path = os.path.join(stat_dir, 'accuracy_hist_norm.png')
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

    hist_path = os.path.join(stat_dir, 'accuracy_hist.png')
    scipy.misc.imsave(hist_path, pred_hist)


    print('total # files: %d' % num_files)
    print('min/max/avg. paths: %d, %d, %.3f' % (min_paths, max_paths, avg_paths))
    print('min/max/avg. abs diff labels: %d, %d, %.3f' % (min_diff_labels, max_diff_labels, avg_diff_labels))
    print('min/max/avg. accuracy: %.3f, %.3f, %.3f' % (min_acc, max_acc, avg_acc))
    print('min/max/avg. duration (sec): %.3f, %.3f, %.3f' % (min_duration, max_duration, avg_duration))
    
    result_path = os.path.join(stat_dir, '_result.txt')
    f = open(result_path, 'w')
    f.write('min/max/avg. paths: %d, %d, %.3f\n' % (min_paths, max_paths, avg_paths))
    f.write('min/max/avg. abs diff labels: %d, %d, %.3f\n' % (min_diff_labels, max_diff_labels, avg_diff_labels))
    f.write('min/max/avg. accuracy: %.3f, %.3f, %.3f\n' % (min_acc, max_acc, avg_acc))
    f.write('min/max/avg. duration (sec): %.3f, %.3f, %.3f\n' % (min_duration, max_duration, avg_duration))    
    f.close()


def test():
    # create managers
    start_time = time.time()
    print('%s: manager loading...' % datetime.now())

    if FLAGS.use_batch:
        dist = int(FLAGS.window_size * FLAGS.neighbor_sigma + 0.5)
        crop_size = 2 * dist + 1
        # linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width], crop_size)
        linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width])
    else:
        linenet_manager = LinenetManager([FLAGS.image_height, FLAGS.image_width])
    duration = time.time() - start_time
    print('%s: manager loaded (%.3f sec)' % (datetime.now(), duration))

    start_time = time.time()
    print('%s: intersect manager loading...' % datetime.now())
    intersectnet_manager = IntersectnetManager([FLAGS.image_height, FLAGS.image_width])
    duration = time.time() - start_time
    print('%s: manager loaded (%.3f sec)' % (datetime.now(), duration))
    
    stat_path = os.path.join(FLAGS.test_dir, 'stat.txt')
    sf = open(stat_path, 'w')

    num_files = 0
    file_path_list = []
        
    if FLAGS.file_list:
        file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
        with open(file_list_path, 'r') as f:
            while True:
                line = f.readline()
                if not line: break

                file = line.rstrip()
                file_path = os.path.join(FLAGS.data_dir, file)
                file_path_list.append(file_path)
    else:
        for root, _, files in os.walk(FLAGS.data_dir):
            for file in files:
                if not file.lower().endswith('svg_pre'): # 'png'):
                    continue

                file_path = os.path.join(FLAGS.data_dir, file)
                file_path_list.append(file_path)

    file_path_list.sort()
    num_total_test_files = len(file_path_list)
    FLAGS.num_test_files = min(num_total_test_files, FLAGS.num_test_files)
    np.random.seed(0)
    file_path_list_id = np.random.choice(num_total_test_files, FLAGS.num_test_files)
    file_path_list_id.sort()

    for file_path_id in file_path_list_id:
        file_path = file_path_list[file_path_id]
        start_time = time.time()
        num_files += 1
        num_labels, diff_labels, acc_avg = graphcut(linenet_manager, intersectnet_manager, file_path)
        duration = time.time() - start_time
        print('%s:%d/%d-%s processed (%.3f sec)' % (datetime.now(), num_files, FLAGS.num_test_files, file, duration))
        sf.write('%s %d %d %.3f %.3f\n' % (file_path.split('/')[-1], num_labels, diff_labels, acc_avg, duration))
    
    sf.close()
    postprocess(FLAGS.test_dir)
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

    # # pp only
    # postprocess(FLAGS.test_dir)

    # create test directory
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)
    
    # start
    test()


if __name__ == '__main__':
    tf.app.run()