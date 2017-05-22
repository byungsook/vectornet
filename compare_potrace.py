from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import argparse
from datetime import datetime

import cairosvg
from PIL import Image
import io
import numpy as np
import scipy.misc
from subprocess import call
import shutil

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import xml.etree.ElementTree as et

# flags
def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                    default='data/line_ov', # 'data_tmp/gc_test',
                    help='data directory',
                    nargs='?') # optional arg.
    parser.add_argument('dst_dir',
                    default='result/compare/potrace/line', # 'data_tmp/gc_test',
                    help='destination directory',
                    nargs='?') # optional arg.
    parser.add_argument('file_list',
                    default='test.txt',
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('num_test_files',
                    default=100,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('image_width',
                    default=64,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('image_height',
                    default=64,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('potrace_dir',
                    default='result/compare/potrace/bin',
                    help='potrace directory',
                    nargs='?') # optional arg.
    parser.add_argument('use_mp',
                    default=False,
                    help='multiprocessing',
                    nargs='?') # optional arg.

    return parser.parse_args()

FLAGS = init_arg_parser()


def compare_potrace():
    num_files = 0
    file_path_list = []
    
    # file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
    # with open(file_list_path, 'r') as f:
    #     while True:
    #         line = f.readline()
    #         if not line: break

    #         file = line.rstrip()
    #         file_path = os.path.join(FLAGS.data_dir, file)
    #         file_path_list.append(file_path)

    # read entire svg
    for root, _, files in os.walk(FLAGS.data_dir):
        for file in files:
            if not file.lower().endswith('svg'):
                continue
            
            file_path = os.path.join(FLAGS.data_dir, file)
            file_path_list.append(file_path)
            # file_name = file.split('_')[0]
            # file_path = os.path.join(FLAGS.data_dir, file_name+'.svg_pre')
            # file_path_list.append(file_path)

    # select test files
    num_total_test_files = len(file_path_list)
    FLAGS.num_test_files = min(num_total_test_files, FLAGS.num_test_files)
    # np.random.seed(0)
    # file_path_list_id = np.random.choice(num_total_test_files, FLAGS.num_test_files)
    # file_path_list.sort()
    file_path_list_id = xrange(FLAGS.num_test_files)

    # run with multiprocessing
    if FLAGS.use_mp:
        queue = multiprocessing.JoinableQueue()
        num_cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cpus, vectorize_mp, (queue,))


    for file_path_id in file_path_list_id:
        file_path = file_path_list[file_path_id]

        if FLAGS.use_mp:
            queue.put(file_path)
        else:
            vectorize(file_path)

    if FLAGS.use_mp:
        queue.join()
        pool.terminate()
        pool.join()

    # compute accuracy
    acc_avg_total = 0.0
    for file_path_id in file_path_list_id:
        file_path = file_path_list[file_path_id]

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        svg_file_path = os.path.join(FLAGS.dst_dir, file_name+'.svg')

        if not os.path.exists(svg_file_path):
            continue

        acc_avg = compute_accuracy(file_path, svg_file_path)
        print(file_path, 'acc:%.2f' % acc_avg)
        acc_avg_total += acc_avg
        new_file_name = svg_file_path[:-4] + '_%.2f' % acc_avg + svg_file_path[-4:]
        os.rename(svg_file_path, new_file_name)
    acc_avg_total /= FLAGS.num_test_files
    print('acc_avg: %.3f' % acc_avg_total)

    stat_path = os.path.join(FLAGS.dst_dir, 'stat.txt')
    with open(stat_path, 'w') as f:
        f.write('acc_avg: %.3f' % acc_avg_total)


def vectorize_mp(queue):
    while True:
        file_path = queue.get()
        if file_path is None:
            break

        vectorize(file_path)
        queue.task_done()


def vectorize(file_path):
    run_potrace(svgpre2bmp(file_path))


def svgpre2bmp(file_path):
    print(file_path)
    
    f = open(file_path, 'r')
    try:
        svg = f.read()
    except:
        f.close()
        f = open(file_path, 'r', encoding='utf8')
        svg = f.read()
    f.close()


    # ########
    # #### ch1, ch2
    # num_path = svg.count('path d')
    # if num_path == 0:
    #     # c2
    #     num_path = svg.count('path id')
    #     r = 0
    #     s = [1, 1] 
    #     t = [0, 0] 
    # else:
    #     # c1
    #     r = 0
    #     s = [1, -1]
    #     t = [0, -900]
    # svg = svg.format(
    #         w=FLAGS.image_width, h=FLAGS.image_height,
    #         r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
    # #### ch1, ch2
    # ########

    s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    s_img = Image.open(io.BytesIO(s_png))
    s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(s)
    s = s / max_intensity
    s = 1.0 - s

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    bmp_file_path = os.path.join(FLAGS.dst_dir, file_name+'.bmp')
    scipy.misc.imsave(bmp_file_path, s)
    return bmp_file_path


def run_potrace(bmp_file_path):
    call([os.path.join(FLAGS.potrace_dir,'potrace'), 
          '-s',
        #   '-a 0', # corner threshold, 0: sharp, default: 1
          bmp_file_path])

    svg_file_path = bmp_file_path[:-3] + 'svg'
    with open(svg_file_path, 'r') as f:
        svg = f.read()

    svg = svg.replace('pt', '')
    num_paths = svg.count('path')

    cmap = plt.get_cmap('jet')
    cnorm  = colors.Normalize(vmin=0, vmax=num_paths-1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    end = 0
    stroke_color = ''' style="fill:rgb({r}, {g}, {b})"'''
    for i in xrange(num_paths):
        rgb = cscalarmap.to_rgba(i)
        r = int(rgb[0]*255)
        g = int(rgb[1]*255)
        b = int(rgb[2]*255)

        start = svg.find('path', end)
        end = svg.find('/>', start)
        svg = svg[:end] + stroke_color.format(r=r,g=g,b=b) + svg[end:]

    with open(svg_file_path, 'w') as f:
        f.write(svg)

    return svg_file_path


def compute_accuracy(file_path, svg_file_path):
    stroke_list = get_stroke_list(file_path)

    acc_id_list = []
    acc_list = []

    path_starts = []
    with open(svg_file_path, 'r') as f:
        svg = f.read()
        num_paths = svg.count('path')

    for i in xrange(num_paths):
        svg_xml = et.fromstring(svg)
        stroke = svg_xml[1][i]
        for c in reversed(xrange(num_paths)):
            if svg_xml[1][c] != stroke:
                svg_xml[1].remove(svg_xml[1][c])
        svg_one_stroke = et.tostring(svg_xml, method='xml')
        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png)).convert('L')
        i_label_map = (np.array(stroke_img) > 0)

        # # debug
        # plt.imshow(i_label_map, cmap=plt.cm.gray)
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
    return np.average(acc_list)


def get_stroke_list(file_path):
    stroke_list = []
    with open(file_path, 'r') as f:
        svg = f.read()

    stroke_list = []
    # num_paths = svg.count('polyline')

    # for i in xrange(1,num_paths+1):
    #     svg_xml = et.fromstring(svg)
    #     # svg_xml[0]._children = [svg_xml[0]._children[i]]
    #     stroke = svg_xml[i]
    #     for c in reversed(xrange(1,num_paths+1)):
    #         if svg_xml[c] != stroke:
    #             svg_xml.remove(svg_xml[c])
    #     svg_one_stroke = et.tostring(svg_xml, method='xml')

    #     stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
    #     stroke_img = Image.open(io.BytesIO(stroke_png))
    #     stroke = (np.array(stroke_img)[:,:,3] > 0)

    #     # # debug
    #     # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
    #     # plt.imshow(stroke_img, cmap=plt.cm.gray)
    #     # plt.show()

    #     stroke_list.append(stroke)

    ####
    # line start
    svg_xml = et.fromstring(svg)
    num_paths = len(svg_xml[0])

    for i in xrange(num_paths):
        svg_xml = et.fromstring(svg)
        stroke = svg_xml[0][i]
        for c in reversed(xrange(num_paths)):
            if svg_xml[0][c] != stroke:
                svg_xml[0].remove(svg_xml[0][c])
        svg_one_stroke = et.tostring(svg_xml, method='xml')

        y_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        y_img = Image.open(io.BytesIO(y_png)).convert('L')
        y = (np.array(y_img) > 0)

        # # debug
        # plt.imshow(y, cmap=plt.cm.gray)
        # plt.show()
        
        stroke_list.append(y)
    # line end
    ####

    # ###
    # ### ch1, ch2
    # chinese1 = False
    # if chinese1:
    #     r = 0
    #     s = [1, -1]
    #     t = [0, -900]
    # else:
    #     r = 0
    #     s = [1, 1]
    #     t = [0, 0]

    # svg = svg.format(
    #     w=FLAGS.image_width, h=FLAGS.image_height,
    #     r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])

    # if chinese1:
    #     svg_xml = et.fromstring(svg)
    #     num_paths = len(svg_xml[0])

    #     for i in xrange(num_paths):
    #         svg_xml = et.fromstring(svg)
    #         stroke = svg_xml[0][i]
    #         for c in reversed(xrange(num_paths)):
    #             if svg_xml[0][c] != stroke:
    #                 svg_xml[0].remove(svg_xml[0][c])
    #         svg_one_stroke = et.tostring(svg_xml, method='xml')

    #         y_png = cairosvg.svg2png(bytestring=svg_one_stroke)
    #         y_img = Image.open(io.BytesIO(y_png))
    #         y = (np.array(y_img)[:,:,3] > 0)

    #         # # debug
    #         # plt.imshow(y, cmap=plt.cm.gray)
    #         # plt.show()

    #         stroke_list.append(y)
    # else:
    #     id = 0
    #     num_paths = 0
    #     while id != -1:
    #         id = svg.find('path id', id + 1)
    #         num_paths = num_paths + 1
    #     num_paths = num_paths - 1 # uncount last one

    #     for i in reversed(xrange(num_paths)):
    #         id = len(svg)
    #         svg_one_stroke = svg
    #         for c in xrange(num_paths):
    #             id = svg_one_stroke.rfind('path id', 0, id)
    #             if c != i:
    #                 id_start = svg_one_stroke.rfind('>', 0, id) + 1
    #                 id_end = svg_one_stroke.find('/>', id_start) + 2
    #                 svg_one_stroke = svg_one_stroke[:id_start] + svg_one_stroke[id_end:]

    #         y_png = cairosvg.svg2png(bytestring=svg_one_stroke.encode('utf-8'))
    #         y_img = Image.open(io.BytesIO(y_png))
    #         y = (np.array(y_img)[:,:,3] > 0)

    #         # # debug
    #         # plt.imshow(y, cmap=plt.cm.gray)
    #         # plt.show()

    #         stroke_list.append(y)
    # ### ch1, ch2
    # ###
    
    return stroke_list



if __name__ == '__main__':
    # if release mode, change current path
    working_path = os.getcwd()
    if not working_path.endswith('vectornet'):
        working_path = os.path.join(working_path, 'vectornet')
        os.chdir(working_path)

    # FLAGS.dst_dir = 'result/overlap/qdraw/cat_128_full'
    # acc_avg_total = 0
    # count = 0
    # for root, _, files in os.walk(FLAGS.dst_dir):
    #     for file in files:
    #         if not file.lower().endswith('svg'):
    #             continue
            
    #         acc_avg = float(file.split('_')[3][:-4])
        
    #         print(file, 'acc:%.2f' % acc_avg)
    #         acc_avg_total += acc_avg
    #         count += 1
    # acc_avg_total /= count
    # print('acc_avg: %.3f' % acc_avg_total)

    # stat_path = os.path.join(FLAGS.dst_dir, 'stat.txt')
    # with open(stat_path, 'w') as f:
    #     f.write('acc_avg: %.3f' % acc_avg_total)

    if os.path.exists(FLAGS.dst_dir):
        shutil.rmtree(FLAGS.dst_dir)
    os.makedirs(FLAGS.dst_dir)   

    compare_potrace()

    print('Done')