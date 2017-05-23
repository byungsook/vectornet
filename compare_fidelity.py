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
import subprocess
import shutil
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import xml.etree.ElementTree as et

# flags
def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',
                default='data/chinese2', # 'data_tmp/gc_test',
                help='data directory',
                nargs='?') # optional arg.
    parser.add_argument('dst_dir',
                    default='result/compare/fidelity/chinese2_1k_thick', # 'data_tmp/gc_test',
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
                    default=1024,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('image_height',
                    default=1024,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('stroke_width',
                    default=50,
                    help='',
                    nargs='?') # optional arg.
    parser.add_argument('fidelity_dir',
                    default='result/compare/fidelity/bin',
                    help='fidelity directory',
                    nargs='?') # optional arg.
    parser.add_argument('use_mp',
                    default=False,
                    help='multiprocessing',
                    nargs='?') # optional arg.

    return parser.parse_args()

FLAGS = init_arg_parser()


def compare_fidelity():
    num_files = 0
    file_path_list = []

    # if 'chinese' in FLAGS.data_dir or \
    #    'fidelity' in FLAGS.data_dir or \
    #    'qdraw' in FLAGS.data_dir:
    #     file_list_path = os.path.join(FLAGS.data_dir, FLAGS.file_list)
    #     with open(file_list_path, 'r') as f:
    #         while True:
    #             line = f.readline()
    #             if not line: break

    #             file = line.rstrip()
    #             file_path = os.path.join(FLAGS.data_dir, file)
    #             file_path_list.append(file_path)
    # elif 'line' in FLAGS.data_dir:
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
        num_cpus = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(num_cpus, vectorize_mp, (queue,))

    acc_avg_total = 0.0
    for file_path_id in file_path_list_id:
        file_path = file_path_list[file_path_id]

        if FLAGS.use_mp:
            queue.put(file_path)
        else:
            vectorize(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            svg_file_path = os.path.join(FLAGS.dst_dir, file_name+'.svg')

            if not os.path.exists(svg_file_path):
                continue
        
            acc_avg = compute_accuracy(file_path, svg_file_path)
            print(file_path, 'acc:%.2f' % acc_avg)
            acc_avg_total += acc_avg
            new_file_name = svg_file_path[:-4] + '_%.2f' % acc_avg + svg_file_path[-4:]
            os.rename(svg_file_path, new_file_name)
        
    if FLAGS.use_mp:
        queue.join()
        pool.terminate()
        pool.join()

        # compute accuracy
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

    target_dir = os.path.join(FLAGS.dst_dir,'error_speed')
    shutil.rmtree(target_dir)

    

def vectorize_mp(queue):
    while True:
        file_path = queue.get()
        if file_path is None:
            break

        vectorize(file_path)
        queue.task_done()


def vectorize(file_path):
    run_fidelity(svgpre2png(file_path))


def svgpre2png(file_path):
    print(file_path)
    
    f = open(file_path, 'r')
    try:
        svg = f.read()
    except:
        f.close()
        f = open(file_path, 'r', encoding='utf8')
        svg = f.read()
    f.close()

    if 'chinese' in FLAGS.data_dir:
        ########
        #### ch1, ch2
        num_path = svg.count('path d')
        if num_path == 0:
            # c2
            num_path = svg.count('path id')
            r = 0
            s = [1, 1] 
            t = [0, 0] 
        else:
            # c1
            r = 0
            s = [1, -1]
            t = [0, -900]
        svg = svg.format(
                w=FLAGS.image_width, h=FLAGS.image_height,
                r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        #### ch1, ch2
        ########
    elif 'line' in FLAGS.data_dir:
        start = svg.find('width')
        end = svg.find('xmlns', start) - 1
        svg = svg[:start] + 'width="%d" height="%d" viewBox="0 0 64 64"' % (
                FLAGS.image_width, FLAGS.image_height) + svg[end:]
    elif 'fidelity' in FLAGS.data_dir:
        svg = svg.format(w=FLAGS.image_width, h=FLAGS.image_height)

    s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    s_img = Image.open(io.BytesIO(s_png))
    s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(s)
    s = s / max_intensity
    s = 1.0 - s

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    img_file_path = os.path.join(FLAGS.dst_dir, file_name+'.png')
    scipy.misc.imsave(img_file_path, s)
    
    return img_file_path


def run_fidelity(img_file_path):
    file_name = os.path.splitext(os.path.basename(img_file_path))[0]
    dir_path = os.path.join(FLAGS.dst_dir, file_name)

    # copy bin
    if not os.path.exists(dir_path):
        shutil.copytree(FLAGS.fidelity_dir, dir_path)
    
    target_dir = os.path.join(FLAGS.dst_dir,'error_speed')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    conf_dir_path = os.path.join(dir_path, 'config/default_dir_open.conf')
    with open(conf_dir_path, 'w') as f:
        f.write(os.path.join(os.getcwd(),FLAGS.dst_dir))

    conf_sketch_path = os.path.join(dir_path, 'config/input_sketch.conf')
    with open(conf_sketch_path, 'w') as f:
        f.write(os.path.join(os.getcwd(),FLAGS.dst_dir)+'\n')
        f.write('1\n')
        f.write(file_name+'.png')

    current_path = os.getcwd()
    os.chdir(dir_path)
    try:
        subprocess.check_call(['project2_no_gui.exe', 
            '0', # thickness
            '0.5', # lambda - ratio between u_fidelity and u_simplicity, full fidelity
            '0.9995', # dt - the decrease speed of the simulated annealing
            '1', # test start num
            '2']) # test end num 
    except subprocess.CalledProcessError as e:
        # print(e.output) # handle errors in the called executable
        pass
    except OSError:
        pass # executable not found

    os.chdir(current_path)
    shutil.rmtree(dir_path)

    svg_file_path = None
    for root, _, files in os.walk(target_dir):
        for file in files:
            if not file.lower().endswith('svg'):
                os.remove(os.path.join(target_dir,file))
                continue

            if file_name+'_w' in file:
                svg_file_path = os.path.join(FLAGS.dst_dir,file_name+'.svg')
                os.rename(os.path.join(target_dir,file), svg_file_path)
                break

    if svg_file_path is not None:
        # wh: 128 -> FLAGS.image_width, height
        with open(svg_file_path, 'r') as f:
            svg = f.read()
            start = svg.find('height')
            end = svg.find('>', start)
            svg = svg[:start] + 'height="%d" width="%d" viewBox="0 0 %d %d"' % (
                FLAGS.image_height, FLAGS.image_width,
                FLAGS.image_height*2, FLAGS.image_width*2) + svg[end:]

            num_paths = svg.count('path')
            end = 0
            for i in xrange(num_paths):
                start = svg.find('<path', end)
                end = svg.find('>', start)+2
                if 'nan' in svg[start:end]:
                    svg = svg[:start] + svg[end:]
                    end = start
                else:
                    w_start = svg.find('stroke-width', start)
                    w_end = svg.find('fill', w_start)
                    svg = svg[:w_start] + 'stroke-width="%d" ' % FLAGS.stroke_width + svg[w_end:]                    

        with open(svg_file_path, 'w') as f:
            f.write(svg)
    else:
        print(img_file_path)

    return svg_file_path


def compute_accuracy(file_path, svg_file_path):
    stroke_list = get_stroke_list(file_path)

    acc_id_list = []
    acc_list = []

    path_starts = []
    with open(svg_file_path, 'r') as f:
        svg = f.read()

    num_paths = svg.count('path')
    if num_paths == 0:
        return 0

    # end = 0
    # for i in xrange(num_paths):
    #     start = svg.find('<path', end)
    #     end = svg.find('>', start)+2
    #     if 'nan' in svg[start:end]:
    #         svg = svg[:start] + svg[end:]
    #         end = start

    # num_paths = svg.count('path')
    # if num_paths == 0:
    #     return 0

    for i in xrange(num_paths):
        svg_xml = et.fromstring(svg)
        stroke = svg_xml[i]
        for c in reversed(xrange(num_paths)):
            if svg_xml[c] != stroke:
                svg_xml.remove(svg_xml[c])
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

            # # debug
            # plt.figure()
            # plt.subplot(131)
            # plt.imshow(i_label_map, cmap=plt.cm.gray)
            # plt.subplot(132)
            # plt.imshow(np.logical_and(i_label_map, stroke), cmap=plt.cm.gray)
            # plt.subplot(133)
            # plt.imshow(np.logical_or(i_label_map, stroke), cmap=plt.cm.gray)
            # plt.show()

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
    f = open(file_path, 'r')
    try:
        svg = f.read()
    except:
        f.close()
        f = open(file_path, 'r', encoding='utf8')
        svg = f.read()
    f.close()

    if 'chinese' in FLAGS.data_dir:
        ###
        ### ch1, ch2
        num_path = svg.count('path d')
        chinese1 = (num_path != 0)

        if chinese1:
            r = 0
            s = [1, -1]
            t = [0, -900]
        else:
            r = 0
            s = [1, 1]
            t = [0, 0]

        svg = svg.format(
            w=FLAGS.image_width, h=FLAGS.image_height,
            r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])

        if chinese1:
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
                y_img = Image.open(io.BytesIO(y_png))
                y = (np.array(y_img)[:,:,3] > 0)

                # # debug
                # plt.imshow(y, cmap=plt.cm.gray)
                # plt.show()

                stroke_list.append(y)
        else:
            id = 0
            num_paths = 0
            while id != -1:
                id = svg.find('path id', id + 1)
                num_paths = num_paths + 1
            num_paths = num_paths - 1 # uncount last one

            for i in reversed(xrange(num_paths)):
                id = len(svg)
                svg_one_stroke = svg
                for c in xrange(num_paths):
                    id = svg_one_stroke.rfind('path id', 0, id)
                    if c != i:
                        id_start = svg_one_stroke.rfind('>', 0, id) + 1
                        id_end = svg_one_stroke.find('/>', id_start) + 2
                        svg_one_stroke = svg_one_stroke[:id_start] + svg_one_stroke[id_end:]

                y_png = cairosvg.svg2png(bytestring=svg_one_stroke.encode('utf-8'))
                y_img = Image.open(io.BytesIO(y_png))
                y = (np.array(y_img)[:,:,3] > 0)

                # # debug
                # plt.imshow(y, cmap=plt.cm.gray)
                # plt.show()

                stroke_list.append(y)
        ### ch1, ch2
        ###
    elif 'line' in FLAGS.data_dir:
        ####
        # line start
        start = svg.find('width')
        end = svg.find('xmlns', start) - 1
        svg = svg[:start] + 'width="%d" height="%d" viewBox="0 0 64 64"' % (
                FLAGS.image_width, FLAGS.image_height) + svg[end:]
                
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
    elif 'fidelity' in FLAGS.data_dir:
        svg = svg.format(w=FLAGS.image_width, h=FLAGS.image_height)
        
        num_paths = svg.count('<path')
        path_list = []
        end = 0
        for i in xrange(num_paths):
            start = svg.find('<path', end)
            end = svg.find('/>', start) + 2
            path_list.append([start,end])

        stroke_list = []
        for path_id in xrange(num_paths):
            y_svg = svg[:path_list[0][0]] + svg[path_list[path_id][0]:path_list[path_id][1]] + svg[path_list[-1][1]:]
            y_png = cairosvg.svg2png(bytestring=y_svg.encode('utf-8'))
            y_img = Image.open(io.BytesIO(y_png))
            y = np.array(y_img)[:,:,3].astype(np.float)
            stroke_list.append(y)
    elif 'qdraw' in FLAGS.data_dir:
        num_paths = svg.count('polyline')

        for i in xrange(1,num_paths+1):
            svg_xml = et.fromstring(svg)
            # svg_xml[0]._children = [svg_xml[0]._children[i]]
            stroke = svg_xml[i]
            for c in reversed(xrange(1,num_paths+1)):
                if svg_xml[c] != stroke:
                    svg_xml.remove(svg_xml[c])
            svg_one_stroke = et.tostring(svg_xml, method='xml')

            stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
            stroke_img = Image.open(io.BytesIO(stroke_png))
            stroke = (np.array(stroke_img)[:,:,3] > 0)
            stroke_list.append(stroke)

    return stroke_list


if __name__ == '__main__':
    # if release mode, change current path
    working_path = os.getcwd()
    if not working_path.endswith('vectornet'):
        working_path = os.path.join(working_path, 'vectornet')
        os.chdir(working_path)

    # acc_avg_total = 0
    # for root, _, files in os.walk(FLAGS.dst_dir):
    #     for file in files:
    #         if not file.lower().endswith('svg'):
    #             continue
            
    #         acc_avg = float(file.split('_')[1][:-4])
        
    #         print(file, 'acc:%.2f' % acc_avg)
    #         acc_avg_total += acc_avg
    # acc_avg_total /= 99.0
    # print('acc_avg: %.3f' % acc_avg_total)

    # stat_path = os.path.join(FLAGS.dst_dir, 'stat.txt')
    # with open(stat_path, 'w') as f:
    #     f.write('acc_avg: %.3f' % acc_avg_total)

    num_test_files = 100
    # #######
    # # ch1, 64
    # FLAGS.stroke_width = 4
    # FLAGS.image_height = 64
    # FLAGS.image_width = 64
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/chinese1'
    # FLAGS.dst_dir = 'result/compare/fidelity/chinese1_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ###

    # #######
    # # ch2, 64
    # FLAGS.stroke_width = 4
    # FLAGS.image_height = 64
    # FLAGS.image_width = 64
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/chinese2'
    # FLAGS.dst_dir = 'result/compare/fidelity/chinese2_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ###

    # ########
    # ## line_ov, 64
    # FLAGS.stroke_width = 4
    # FLAGS.image_height = 64
    # FLAGS.image_width = 64
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/line_ov'
    # FLAGS.dst_dir = 'result/compare/fidelity/line_ov_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####

    # #######
    # # ch1, 1k
    # FLAGS.stroke_width = 60
    # FLAGS.image_height = 1024
    # FLAGS.image_width = 1024
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/chinese1'
    # FLAGS.dst_dir = 'result/compare/fidelity/chinese1_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ###

    # #######
    # # ch2, 1k
    # FLAGS.stroke_width = 60
    # FLAGS.image_height = 1024
    # FLAGS.image_width = 1024
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/chinese2'
    # FLAGS.dst_dir = 'result/compare/fidelity/chinese2_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ###

    # ########
    # ## line_ov, 1k
    # FLAGS.stroke_width = 60
    # FLAGS.image_height = 1024
    # FLAGS.image_width = 1024
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/line_ov'
    # FLAGS.dst_dir = 'result/compare/fidelity/line_ov_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####

    # ########
    # ## baseball, 128
    # FLAGS.stroke_width = 6
    # FLAGS.image_height = 128
    # FLAGS.image_width = 128
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/qdraw/qdraw_baseball_128'
    # FLAGS.dst_dir = 'result/compare/fidelity/qdraw/baseball_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####

    # ########
    # ## cat, 128
    # FLAGS.stroke_width = 6
    # FLAGS.image_height = 128
    # FLAGS.image_width = 128
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/qdraw/qdraw_cat_128'
    # FLAGS.dst_dir = 'result/compare/fidelity/qdraw/cat_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####
    # ########
    # ## stitches, 128
    # FLAGS.stroke_width = 6
    # FLAGS.image_height = 128
    # FLAGS.image_width = 128
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/qdraw/qdraw_stitches_128'
    # FLAGS.dst_dir = 'result/compare/fidelity/qdraw/stitches_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####

    # ########
    # ## chandelier, 128
    # FLAGS.stroke_width = 6
    # FLAGS.image_height = 128
    # FLAGS.image_width = 128
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/qdraw_chandelier_128_test'
    # FLAGS.dst_dir = 'result/compare/fidelity/qdraw/chandelier_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####

    # ########
    # ## elephant, 128
    # FLAGS.stroke_width = 6
    # FLAGS.image_height = 128
    # FLAGS.image_width = 128
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/qdraw_elephant_128_test'
    # FLAGS.dst_dir = 'result/compare/fidelity/qdraw/elephant_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####

    # ########
    # ## backpack, 128
    # FLAGS.stroke_width = 6
    # FLAGS.image_height = 128
    # FLAGS.image_width = 128
    # FLAGS.num_test_files = num_test_files
    # FLAGS.data_dir = 'data/qdraw_backpack_128'
    # FLAGS.dst_dir = 'result/compare/fidelity/qdraw/backpack_%d_%d' % (
    #                  FLAGS.image_height, FLAGS.stroke_width)
    # if os.path.exists(FLAGS.dst_dir):
    #     shutil.rmtree(FLAGS.dst_dir)
    # os.makedirs(FLAGS.dst_dir)       
    # compare_fidelity()
    # ####

    ########
    ## mix, 128
    FLAGS.stroke_width = 6
    FLAGS.image_height = 128
    FLAGS.image_width = 128
    FLAGS.num_test_files = num_test_files
    FLAGS.data_dir = 'data/qdraw_bicycle_128'
    FLAGS.dst_dir = 'result/compare/fidelity/qdraw/bicycle_%d_%d' % (
                     FLAGS.image_height, FLAGS.stroke_width)
    if os.path.exists(FLAGS.dst_dir):
        shutil.rmtree(FLAGS.dst_dir)
    os.makedirs(FLAGS.dst_dir)       
    compare_fidelity()
    ####

    print('Done')