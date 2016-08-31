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
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from numpy import linalg as LA
import scipy.stats
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
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
tf.app.flags.DEFINE_float('neighbor_sigma', 5.0,
                           """neighbor sigma""")
tf.app.flags.DEFINE_float('neighbor_power', 2.0,
                           """neighbor power""")
tf.app.flags.DEFINE_float('neighbor_weight', 100.0,
                           """make it sensible value when rounding""")
tf.app.flags.DEFINE_integer('max_num_labels', 5, 
                           """the maximum number of labels""")


def _imread(img_file_name, inv=False):
    """ Read, grayscale and normalize the image"""
    img = np.array(Image.open(img_file_name).convert('L')).astype(np.float) / 255.0
    if inv: 
        return 1.0 - img
    else: 
        return img

def graphcut(linenet_manager, file_path):
    img = _imread(file_path, inv=True)

    # # debug
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()

    # compute probability map of all line pixels
    y_batch, line_pixels = linenet_manager.extract_all(img)
    
    # specify neighbor weights
    num_line_pixels = len(line_pixels[0])

    ###################################################################################
    # # debug: generate similarity map
    # for i in xrange(num_line_pixels):
    #     p1 = np.array([line_pixels[0][i], line_pixels[1][i]])
    #     similarity_map = np.zeros([FLAGS.image_size, FLAGS.image_size, 3], dtype=np.float)
        
    #     for j in xrange(num_line_pixels):
    #         if i == j:
    #             continue
    #         p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
    #         d12 = LA.norm(p1-p2, 2)
    #         w12 = scipy.stats.norm(0, FLAGS.neighbor_sigma).pdf(d12)
    #         y_diff = np.reshape(y_batch[i,:,:,:] - y_batch[j,:,:,:], [FLAGS.image_size, FLAGS.image_size])
    #         norm12 = LA.norm(y_diff, 'fro')
    #         similarity = (w12 / norm12) ** FLAGS.neighbor_power
    #         # print('similarity', similarity)
    #         similarity_map[p2[0],p2[1]] = np.array([similarity, similarity, similarity])

    #     # print('max similiarity', np.amax(similarity_map))
    #     similarity_map = similarity_map / np.amax(similarity_map)
    #     similarity_map[p1[0],p1[1]] = np.array([1, 0, 0])
    #     # plt.imshow(similarity_map)
    #     # plt.show()
    #     save_path = os.path.join(FLAGS.test_dir, 'similarity_map_%d.png' % i)
    #     scipy.misc.imsave(save_path, similarity_map)

    # print('Done')
    # return
    ###################################################################################

    # support only symmetric edge weight
    edge_weight = []
    for i in xrange(num_line_pixels-1):
        p1 = np.array([line_pixels[0][i], line_pixels[1][i]])        
        for j in xrange(i+1, num_line_pixels):
            p2 = np.array([line_pixels[0][j], line_pixels[1][j]])
            d12 = LA.norm(p1-p2, 2)
            w12 = scipy.stats.norm(0, FLAGS.neighbor_sigma).pdf(d12)
            y_diff = np.reshape(y_batch[i,:,:,:] - y_batch[j,:,:,:], [FLAGS.image_size, FLAGS.image_size])
            norm12 = LA.norm(y_diff, 'fro')
            similarity = int(((w12 / norm12) ** FLAGS.neighbor_power) * FLAGS.neighbor_weight)
            # print('similarity', similarity)
            if similarity > 0:
                edge_weight.append([i, j, similarity])
    
    edge_weight = np.array(edge_weight).astype(np.int32)

    # graphcut opt.
    data_term = np.ones([num_line_pixels*num_line_pixels, FLAGS.max_num_labels], dtype=np.int32)
    pairwise = np.ones([FLAGS.max_num_labels, FLAGS.max_num_labels], dtype=np.int32) - np.eye(FLAGS.max_num_labels, dtype=np.int32)
    result_label = pygco.cut_from_graph(edge_weight, data_term, pairwise)
    print('the number of labels', np.amax(result_label))
        
    label_map = np.ones([FLAGS.image_size, FLAGS.image_size, 3]) * 255
    for i in xrange(num_line_pixels):
        np.random.seed(result_label[i])
        color = np.random.randint(0, high=256, size=3)
        label_map[line_pixels[0][i],line_pixels[1][i]] = color

    label_map_path = os.path.join(FLAGS.test_dir, 'label_map.png')
    scipy.misc.imsave(label_map_path, label_map)


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
            
            # debug
            return

    print('Done')


def main(_):
    # if release mode, change current path
    current_path = os.getcwd()
    if not current_path.endswith('vectornet'):
        working_path = os.path.join(current_path, 'vectornet')
        os.chdir(working_path)
    
    # create test directory
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)

    # start
    test()


if __name__ == '__main__':
    tf.app.run()