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
    with open(file_path, 'r') as sf:
        svg = sf.read().format(w=FLAGS.image_width, h=FLAGS.image_height, sw=10,
                               bx=0, by=0, bw=800, bh=800)

    s_png = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    s_img = Image.open(io.BytesIO(s_png))
    s = np.array(s_img)[:,:,3].astype(np.float) # / 255.0
    max_intensity = np.amax(s)
    s = s / max_intensity    

    # # debug
    # plt.imshow(s, cmap=plt.cm.gray)
    # plt.show()

    # leave only one path
    svg_xml = ET.fromstring(svg)
    num_paths = len(svg_xml[0])    
    return s, num_paths


def get_stroke_list(pm):
    stroke_list = []
    with open(pm.file_path, 'r') as sf:
        svg = sf.read().format(w=FLAGS.image_width, h=FLAGS.image_height, sw=10,
                                bx=0, by=0, bw=800, bh=800)
        
    stroke_list = []
    svg_xml = ET.fromstring(svg)
    num_paths = len(svg_xml[0])

    for i in xrange(num_paths):
        svg_xml = ET.fromstring(svg)
        stroke = svg_xml[0][i]
        for c in reversed(xrange(num_paths)):
            if svg_xml[0][c] != stroke:
                svg_xml[0].remove(svg_xml[0][c])
        svg_one_stroke = ET.tostring(svg_xml, method='xml')

        stroke_png = cairosvg.svg2png(bytestring=svg_one_stroke)
        stroke_img = Image.open(io.BytesIO(stroke_png))
        stroke = (np.array(stroke_img)[:,:,3] > 0)

        # # debug
        # stroke_img = np.array(stroke_img)[:,:,3].astype(np.float) / 255.0
        # plt.imshow(stroke_img, cmap=plt.cm.gray)
        # plt.show()

        stroke_list.append(stroke)
        
    return stroke_list
