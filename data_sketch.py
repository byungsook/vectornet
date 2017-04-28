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

FLAGS = tf.app.flags.FLAGS


def read_svg(file_path):
    with open(file_path, 'r') as f:
        svg = f.read()
        svg_xml = ET.fromstring(svg)
        num_paths = len(svg_xml[0]) - 1

        r = 0
        s = [1, 1]
        t = [0, 0]
        svg = svg.format(w=FLAGS.image_width, h=FLAGS.image_height, 
                         r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])
        s_png = cairosvg.svg2png(bytestring=svg)
        s_img = Image.open(io.BytesIO(s_png))
        s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity
    return s, num_paths


def get_stroke_list(pm):
    stroke_list = []
    with open(pm.file_path, 'r') as f:
        svg = f.read()
        r = 0
        s = [1, 1]
        t = [0, 0]
        svg = svg.format(w=FLAGS.image_width, h=FLAGS.image_height, 
                         r=r, sx=s[0], sy=s[1], tx=t[0], ty=t[1])

        svg_xml = ET.fromstring(svg)
        num_paths = len(svg_xml[0])

        for i in xrange(1,num_paths):
            svg_xml = ET.fromstring(svg)
            stroke = svg_xml[0][i]
            for c in reversed(xrange(1,num_paths)):
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
        
    return stroke_list
