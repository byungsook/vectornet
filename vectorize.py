# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import multiprocessing
import os
import os.path
import time
from subprocess import call
import pprint

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import scipy.stats
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sklearn.neighbors
import skimage.measure

import tensorflow as tf
from pathnet.pathnet_manager import PathnetManager
from ovnet.ovnet_manager import OvnetManager


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_dir', 'log/test',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'data/line_ov',
                           """Data directory""")
tf.app.flags.DEFINE_string('file_list', 'test.txt',
                           """file_list""")
tf.app.flags.DEFINE_integer('num_test_files', 8,
                           """num_test_files""")
tf.app.flags.DEFINE_integer('image_width', 64,
                            """Image Width.""")
tf.app.flags.DEFINE_integer('image_height', 64,
                            """Image Height.""")
tf.app.flags.DEFINE_integer('max_num_labels', 128,
                           """the maximum number of labels""")
tf.app.flags.DEFINE_integer('label_cost', 0,
                           """label cost""")
tf.app.flags.DEFINE_float('neighbor_sigma', 8, # 48 - 8, 64 - 10.67?
                           """neighbor sigma""")
tf.app.flags.DEFINE_float('prediction_sigma', 0.7,
                           """prediction sigma""")
tf.app.flags.DEFINE_boolean('compile', False,
                            """whether compile gco or not""")
tf.app.flags.DEFINE_boolean('find_overlap', True,
                            """whether to find overlap or not""")
tf.app.flags.DEFINE_string('data_type', 'line',
                           """specify data""")
tf.app.flags.DEFINE_integer('batch_size', 512,
                           """batch size""")


if FLAGS.data_type == 'chinese':
    from data_chinese import read_svg
    from data_chinese import get_stroke_list
elif FLAGS.data_type == 'sketch':
    from data_sketch import read_svg
    from data_sketch import get_stroke_list
elif FLAGS.data_type == 'sketch2':
    from data_sketch2 import read_svg
    from data_sketch2 import get_stroke_list
elif FLAGS.data_type == 'line':
    from data_line import read_svg
    from data_line import get_stroke_list
else:
    print('wrong data set')
    assert(False)


class Param(object):
    def __init__(self):
        self.file_path = None
        self.duration = None
        self.path_pixels = None
        self.dup_dict = None
        self.dup_rev_dict = None
        self.num_labels = None
        self.diff_labels = None
        self.acc_avg = None
        self.img = None


def predict(pathnet_manager, ovnet_manager, file_path):
    # path for temporary files
    tf.gfile.MakeDirs(FLAGS.test_dir + '/tmp')

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print('%s: %s, start vectorization' % (datetime.now(), file_name))

    # convert svg to raster image
    img, num_paths = read_svg(file_path)

    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # predict paths through pathnet
    start_time = time.time()
    y_batch, path_pixels = pathnet_manager.extract_all(img, batch_size=FLAGS.batch_size)
    # # debug
    # plt.imshow(y_batch[0,:,:,0], cmap=plt.cm.gray)
    # plt.show()
    path0_img_path = os.path.join(FLAGS.test_dir, '%s_path0.png' % file_name)
    scipy.misc.imsave(path0_img_path, 1 - y_batch[0,:,:,0])
    num_path_pixels = len(path_pixels[0])
    duration = time.time() - start_time
    print('%s: %s, predict paths (#pixels:%d) through pathnet (%.3f sec)' % (datetime.now(), file_name, num_path_pixels, duration))

    dup_dict = {}
    dup_rev_dict = {}
    dup_id = num_path_pixels # start id of duplicated pixels

    if FLAGS.find_overlap:
        # predict overlap using overlap net
        start_time = time.time()
        overlap = ovnet_manager.overlap(img)

        overlap_img_path = os.path.join(FLAGS.test_dir, '%s_overlap.png' % file_name)
        scipy.misc.imsave(overlap_img_path, 1 - overlap)

        # # debug
        # plt.imshow(overlap, cmap=plt.cm.gray)
        # plt.show()

        for i in xrange(num_path_pixels):
            if overlap[path_pixels[0][i], path_pixels[1][i]]:
                dup_dict[i] = dup_id
                dup_rev_dict[dup_id] = i
                dup_id += 1

        # debug
        # print(dup_dict)
        # print(dup_rev_dict)

        duration = time.time() - start_time
        print('%s: %s, predict overlap (#:%d) through ovnet (%.3f sec)' % (datetime.now(), file_name, dup_id-num_path_pixels, duration))

    # write config file for graphcut
    start_time = time.time()
    pred_file_path = os.path.join(FLAGS.test_dir, 'tmp', file_name + '.pred')
    f = open(pred_file_path, 'w')
    # info
    f.write(pred_file_path + '\n')
    f.write(FLAGS.data_dir + '\n')
    f.write('%d\n' % FLAGS.max_num_labels)
    f.write('%d\n' % FLAGS.label_cost)
    f.write('%f\n' % FLAGS.neighbor_sigma)
    f.write('%f\n' % FLAGS.prediction_sigma)
    # f.write('%d\n' % num_path_pixels)
    f.write('%d\n' % dup_id)

    # # support only symmetric edge weight
    # radius = FLAGS.neighbor_sigma*2
    # nb = sklearn.neighbors.NearestNeighbors(radius=radius)
    # nb.fit(np.array(path_pixels).transpose())

    high_spatial = 1000
    for i in xrange(num_path_pixels-1):
        p1 = np.array([path_pixels[0][i], path_pixels[1][i]])
        pred_p1 = np.reshape(y_batch[i,:,:,:], [FLAGS.image_height, FLAGS.image_width])

        # # see close neighbors
        # rng = nb.radius_neighbors([p1])
        # for rj, j in enumerate(rng[1][0]): # ids
        #     if j <= i:
        #         continue                
        #     p2 = np.array([path_pixels[0][j], path_pixels[1][j]])            
        #     d12 = rng[0][0][rj]

        for j in xrange(i+1, num_path_pixels): # see entire neighbors
            p2 = np.array([path_pixels[0][j], path_pixels[1][j]])
            d12 = np.linalg.norm(p1-p2, 2)
            
            pred_p2 = np.reshape(y_batch[j,:,:,:], [FLAGS.image_height, FLAGS.image_width])
            pred = (pred_p1[p2[0],p2[1]] + pred_p2[p1[0],p1[1]]) * 0.5
            pred = np.exp(-0.5 * (1.0-pred)**2 / FLAGS.prediction_sigma**2)

            spatial = np.exp(-0.5 * d12**2 / FLAGS.neighbor_sigma**2)
            f.write('%d %d %f %f\n' % (i, j, pred, spatial))

            dup_i = dup_dict.get(i)
            if dup_i is not None:
                f.write('%d %d %f %f\n' % (j, dup_i, pred, spatial)) # as dup is always smaller than normal id
                f.write('%d %d %f %f\n' % (i, dup_i, 0, high_spatial)) # shouldn't be labeled together
            dup_j = dup_dict.get(j)
            if dup_j is not None:
                f.write('%d %d %f %f\n' % (i, dup_j, pred, spatial)) # as dup is always smaller than normal id
                f.write('%d %d %f %f\n' % (j, dup_j, 0, high_spatial)) # shouldn't be labeled together

            if dup_i is not None and dup_j is not None:
                f.write('%d %d %f %f\n' % (dup_i, dup_j, pred, spatial)) # dup_i < dup_j

    f.close()
    duration = time.time() - start_time
    print('%s: %s, prediction computed (%.3f sec)' % (datetime.now(), file_name, duration))

    pm = Param()
    pm.num_paths = num_paths
    pm.path_pixels = path_pixels
    pm.dup_dict = dup_dict
    pm.dup_rev_dict = dup_rev_dict
    pm.img = img

    return pm


def vectorize_mp(queue):    
    while True:
        pm = queue.get()
        if pm is None:
            break
        
        vectorize(pm)
        queue.task_done()


def vectorize(pm):
    start_time = time.time()
    file_path = os.path.basename(pm.file_path)
    file_name = os.path.splitext(file_path)[0]

    # 1. label
    labels, e_before, e_after = label(file_name)

    # 2. merge small components
    labels = merge_small_component(labels, pm)
    
    # 2-2. assign one label per one connected component
    # labels = label_cc(labels, pm)

    # 3. compute accuracy
    accuracy_list = compute_accuracy(labels, pm)

    unique_labels = np.unique(labels)
    num_labels = unique_labels.size
    diff_labels = num_labels - pm.num_paths
    acc_avg = np.average(accuracy_list)
    
    print('%s: %s, the number of labels %d, truth %d, diff %d' % (datetime.now(), file_name, num_labels, pm.num_paths, diff_labels))
    print('%s: %s, energy before optimization %.4f' % (datetime.now(), file_name, e_before))
    print('%s: %s, energy after optimization %.4f' % (datetime.now(), file_name, e_after))
    print('%s: %s, accuracy computed, avg.: %.3f' % (datetime.now(), file_name, acc_avg))

    # 4. save image
    save_label_img(labels, unique_labels, num_labels, diff_labels, acc_avg, pm)

    duration = time.time() - start_time
    
    # write result
    pm.duration += duration        
    print('%s: %s, done (%.3f sec)' % (datetime.now(), file_name, pm.duration))
    stat_file_path = os.path.join(FLAGS.test_dir, file_name + '_stat.txt')
    with open(stat_file_path, 'w') as f:
        f.write('%s %d %d %.3f %.3f\n' % (file_path, num_labels, diff_labels, acc_avg, pm.duration))


def label(file_name):
    start_time = time.time()
    working_path = os.getcwd()
    gco_path = os.path.join(working_path, 'gco/qpbo_src')
    os.chdir(gco_path)
    os.environ['LD_LIBRARY_PATH'] = os.getcwd()

    pred_file_path = os.path.join(FLAGS.test_dir, 'tmp', file_name + '.pred')
    if pred_file_path[0] != '/': # relative path
        pred_file_path = '../../' + pred_file_path
    call(['./gco_linenet', pred_file_path])
    os.chdir(working_path)

    # read graphcut result
    label_file_path = os.path.join(FLAGS.test_dir, 'tmp', file_name + '.label')
    f = open(label_file_path, 'r')
    e_before = float(f.readline())
    e_after = float(f.readline())
    labels = np.fromstring(f.read(), dtype=np.int32, sep=' ')
    f.close()
    duration = time.time() - start_time
    print('%s: %s, labeling finished (%.3f sec)' % (datetime.now(), file_name, duration))

    return labels, e_before, e_after


def merge_small_component(labels, pm):
    knb = sklearn.neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knb.fit(np.array(pm.path_pixels).transpose())

    num_path_pixels = len(pm.path_pixels[0])

    for iter in xrange(2):
        # # debug
        # print('%d-th iter' % iter)
        
        unique_label = np.unique(labels)
        for i in unique_label:
            i_label_list = np.nonzero(labels == i)

            # handle duplicated pixels
            for j, i_label in enumerate(i_label_list[0]):
                if i_label >= num_path_pixels:
                    i_label_list[0][j] = pm.dup_rev_dict[i_label]

            # connected component analysis on 'i' label map
            i_label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.float)
            i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = 1.0
            cc_map, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)

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

                    dup = pm.dup_dict.get(indices[0][0])
                    if dup is not None:
                        labels[dup] = max_label_nb

    return labels


def label_cc(labels, pm):
    unique_label = np.unique(labels)
    num_path_pixels = len(pm.path_pixels[0])

    new_label = FLAGS.max_num_labels
    for i in unique_label:
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        # connected component analysis on 'i' label map
        i_label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = 1.0
        cc_map, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)

        if num_cc > 1:
            for i_label in i_label_list[0]:
                cc_label = cc_map[pm.path_pixels[0][i_label],pm.path_pixels[1][i_label]]
                if cc_label > 1:
                    labels[i_label] = new_label + (cc_label-2)

            new_label += (num_cc - 1)

    return labels


def compute_accuracy(labels, pm):
    stroke_list = get_stroke_list(pm)
    
    unique_labels = np.unique(labels)
    num_path_pixels = len(pm.path_pixels[0])

    acc_id_list = []
    acc_list = []
    for i in unique_labels:
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        i_label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.bool)
        i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = True

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


def save_label_img(labels, unique_labels, num_labels, diff_labels, acc_avg, pm):
    file_path = os.path.basename(pm.file_path)
    file_name = os.path.splitext(file_path)[0]
    num_path_pixels = len(pm.path_pixels[0])

    cmap = plt.get_cmap('jet')    
    cnorm = colors.Normalize(vmin=0, vmax=num_labels-1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
    
    label_map = np.ones([FLAGS.image_height, FLAGS.image_width, 3], dtype=np.float)
    label_map_t = np.ones([FLAGS.image_height, FLAGS.image_width, 3], dtype=np.float)
    first_svg = True
    target_svg_path = os.path.join(FLAGS.test_dir, '%s_%d_%d_%.2f.svg' % (file_name, num_labels, diff_labels, acc_avg))
    for color_id, i in enumerate(unique_labels):
        i_label_list = np.nonzero(labels == i)

        # handle duplicated pixels
        for j, i_label in enumerate(i_label_list[0]):
            if i_label >= num_path_pixels:
                i_label_list[0][j] = pm.dup_rev_dict[i_label]

        color = np.asarray(cscalarmap.to_rgba(color_id))
        label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = color[:3]

        # save i label map
        i_label_map = np.zeros([FLAGS.image_height, FLAGS.image_width], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = pm.img[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]]
        _, num_cc = skimage.measure.label(i_label_map, background=0, return_num=True)
        i_label_map_path = os.path.join(FLAGS.test_dir + '/tmp', 'i_%s_%d_%d.bmp' % (file_name, i, num_cc))
        scipy.misc.imsave(i_label_map_path, i_label_map)

        i_label_map = np.ones([FLAGS.image_height, FLAGS.image_width, 3], dtype=np.float)
        i_label_map[pm.path_pixels[0][i_label_list],pm.path_pixels[1][i_label_list]] = color[:3]
        label_map_t += i_label_map

        # vectorize using potrace
        color *= 255
        color_hex = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))
        call(['potrace', '-s', '-i', '-C'+color_hex, i_label_map_path])
        
        i_label_map_svg = os.path.join(FLAGS.test_dir + '/tmp', 'i_%s_%d_%d.svg' % (file_name, i, num_cc))
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

        # remove i label map
        call(['rm', i_label_map_path, i_label_map_svg])

    # set opacity 0.5 to see overlaps
    with open(target_svg_path, 'r') as f:
        target_svg = f.read()
    
    insert_pos = target_svg.find('<g')
    target_svg = target_svg[:insert_pos] + '<g fill-opacity="0.5">' + target_svg[insert_pos:]
    insert_pos = target_svg.find('</svg>')
    target_svg = target_svg[:insert_pos] + '</g>' + target_svg[insert_pos:]
    
    with open(target_svg_path, 'w') as f:
        f.write(target_svg)

    label_map_path = os.path.join(FLAGS.test_dir, '%s_%.2f_%.2f_%d_%d_%.2f.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma, num_labels, diff_labels, acc_avg))
    scipy.misc.imsave(label_map_path, label_map)

    label_map_t /= np.amax(label_map_t)
    label_map_path = os.path.join(FLAGS.test_dir, '%s_%.2f_%.2f_%d_%d_%.2f_t.png' % (file_name, FLAGS.neighbor_sigma, FLAGS.prediction_sigma, num_labels, diff_labels, acc_avg))
    scipy.misc.imsave(label_map_path, label_map_t)


def test():
    # print flags
    flag_file_path = os.path.join(FLAGS.test_dir, 'flag.txt')
    with open(flag_file_path, 'wt') as out:
        pprint.PrettyPrinter(stream=out).pprint(FLAGS.__flags)
        
    # create managers
    start_time = time.time()
    print('%s: pathnet manager loading...' % datetime.now())

    pathnet_manager = PathnetManager([FLAGS.image_height, FLAGS.image_width])
    duration = time.time() - start_time
    print('%s: pathnet manager loaded (%.3f sec)' % (datetime.now(), duration))

    start_time = time.time()
    print('%s: ovnet manager loading...' % datetime.now())
    ovnet_manager = OvnetManager([FLAGS.image_height, FLAGS.image_width])
    duration = time.time() - start_time
    print('%s: ovnet manager loaded (%.3f sec)' % (datetime.now(), duration))
    
    # run with multiprocessing
    queue = multiprocessing.JoinableQueue()
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cpus, vectorize_mp, (queue,))

    if FLAGS.data_type != 'line':
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

        # select test files
        num_total_test_files = len(file_path_list)
        FLAGS.num_test_files = min(num_total_test_files, FLAGS.num_test_files)
        # np.random.seed(0)
        # file_path_list_id = np.random.choice(num_total_test_files, FLAGS.num_test_files)
        # file_path_list.sort()
        file_path_list_id = xrange(FLAGS.num_test_files)

        grand_start_time = time.time()
        for file_path_id in file_path_list_id:
            file_path = file_path_list[file_path_id]
            start_time = time.time()
            # only prediction done by single process because of large network
            pm = predict(pathnet_manager, ovnet_manager, file_path)
            duration = time.time() - start_time

            pm.file_path = file_path
            pm.duration = duration
            queue.put(pm)
            
            # debug
            # vectorize(pm)
    else:
        grand_start_time = time.time()
        for svg_id in xrange(FLAGS.num_test_files):
            start_time = time.time()
            file_path = os.path.join(FLAGS.data_dir, '%d.svg' % svg_id)
            # only prediction done by single process because of large network
            pm = predict(pathnet_manager, ovnet_manager, file_path)
            duration = time.time() - start_time

            pm.file_path = file_path
            pm.duration = duration
            queue.put(pm)

            # debug
            # vectorize(pm)

    queue.join()
    pool.terminate()
    pool.join()
    duration = time.time() - grand_start_time

    # postprocess(FLAGS.test_dir)

    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Done ({:0>2}:{:0>2}:{:05.2f})'.format(int(hours),int(minutes),seconds))


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


if __name__ == '__main__':
    tf.app.run()
