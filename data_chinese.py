# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

import cairosvg
import io
from PIL import Image

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf


# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('chinese1', True,
                            """whether chinese1 or not""")


def read_svg(file_path):
    with open(file_path, 'r') as f:
        svg = f.read()
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
        s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity
    return s, num_path


def get_stroke_list(pm):
    stroke_list = []
    with open(pm.file_path, 'r') as f:
        svg = f.read()
        if FLAGS.chinese1:
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

        if FLAGS.chinese1:
            svg_xml = ET.fromstring(svg)
            num_paths = len(svg_xml[0])

            for i in xrange(num_paths):
                svg_xml = ET.fromstring(svg)
                stroke = svg_xml[0][i]
                for c in reversed(xrange(num_paths)):
                    if svg_xml[0][c] != stroke:
                        svg_xml[0].remove(svg_xml[0][c])
                svg_one_stroke = ET.tostring(svg_xml, method='xml')

                y_png = cairosvg.svg2png(bytestring=svg_one_stroke)
                y_img = Image.open(io.BytesIO(y_png))
                y = (np.array(y_img)[:,:,3] > 0)

                # # debug
                # y_img = np.array(y_img)[:,:,3].astype(np.float) / 255.0
                # plt.imshow(y_img, cmap=plt.cm.gray)
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
                # y_img = np.array(y_img)[:,:,3].astype(np.float) / 255.0
                # plt.imshow(y_img, cmap=plt.cm.gray)
                # plt.show()

                stroke_list.append(y)

    
    return stroke_list
