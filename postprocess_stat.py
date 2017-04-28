# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


parser = argparse.ArgumentParser(description='postprocess stat')
parser.add_argument('--stat_dir', metavar='x', type=str, nargs='?',
                    default='result/no_overlap/bicycle_tr')    
args = parser.parse_args()


def postprocess(stat_dir):
    print('src:', stat_dir)
    num_files = 0
    path_list = []
    diff_list = []
    acc_list = []
    duration_list = []
   
    stat_path = os.path.join(stat_dir, '*_stat.txt')
    stat_list = glob.glob(stat_path)
    
    for i, stat_file_path in enumerate(stat_list):
        with open(stat_file_path, 'r') as f:
            line = f.readline()
            name, num_labels, diff_labels, accuracy, duration = line.split()
            
    # for root, _, files in os.walk(FLAGS.test_dir):
    #     for file in files:
    #         ss = file.split('_')
    #         if len(ss) < 7: continue
    #         name = ss[2]
    #         num_labels = ss[5]
    #         diff_labels = ss[6]
    #         accuracy = ss[7]
    #         accuracy = accuracy.rstrip('.png')
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
    
    bins = max_diff_labels - min_diff_labels + 1    
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
    
    result_path = os.path.join(stat_dir, 'result.txt')
    f = open(result_path, 'w')
    f.write('min/max/avg. paths: %d, %d, %.3f\n' % (min_paths, max_paths, avg_paths))
    f.write('min/max/avg. abs diff labels: %d, %d, %.3f\n' % (min_diff_labels, max_diff_labels, avg_diff_labels))
    f.write('min/max/avg. accuracy: %.3f, %.3f, %.3f\n' % (min_acc, max_acc, avg_acc))
    f.write('min/max/avg. duration (sec): %.3f, %.3f, %.3f\n' % (min_duration, max_duration, avg_duration))    
    f.close()


def main():
    # if release mode, change current path
    working_path = os.getcwd()
    if not working_path.endswith('vectornet'):
        working_path = os.path.join(working_path, 'vectornet')
        os.chdir(working_path)
    
    postprocess(args.stat_dir)


if __name__ == '__main__':
    main()
