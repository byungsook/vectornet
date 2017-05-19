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
from subprocess import call

import cairosvg
import io
from PIL import Image

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_svg(file_path):
    with open(file_path, 'r') as sf:
        svg = sf.read().format(w=FLAGS.image_width, h=FLAGS.image_height)
    num_paths = svg.count('<path')
    img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    img = Image.open(io.BytesIO(img))
    s = np.array(img)[:,:,3].astype(np.float) # / 255.0
    s /= 255.0
    return s, num_paths


def get_stroke_list(pm):
    stroke_list = []
    with open(pm.file_path, 'r') as f:
        svg = f.read().format(w=FLAGS.image_width, h=FLAGS.image_height)
    
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

    return stroke_list
