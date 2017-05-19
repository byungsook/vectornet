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
        svg = sf.read()
        img = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(img))
        s = np.array(img)[:,:,3].astype(np.float) # / 255.0
        max_intensity = np.amax(s)
        s = s / max_intensity
    return s, svg.count('polyline')


def get_stroke_list(pm):
    stroke_list = []
    with open(pm.file_path, 'r') as f:
        svg = f.read()
        num_paths = svg.count('polyline')
        for i in xrange(1,num_paths+1):
            svg_xml = ET.fromstring(svg)
            stroke = svg_xml[i]
            for c in reversed(xrange(1,num_paths+1)):
                if svg_xml[c] != stroke:
                    svg_xml.remove(svg_xml[c])
            svg_one_stroke = ET.tostring(svg_xml, method='xml')

            y_png = cairosvg.svg2png(bytestring=svg_one_stroke)
            y_img = Image.open(io.BytesIO(y_png))
            y = (np.array(y_img)[:,:,3] > 0)

            # # debug
            # y_img = np.array(y_img)[:,:,3].astype(np.float) / 255.0
            # plt.imshow(y_img, cmap=plt.cm.gray)
            # plt.show()

            stroke_list.append(y)

    # call(['rm', pm.file_path])
    return stroke_list

# class Param():
#     pass
# pm = Param()
# pm.file_path = '/home/kimby/dev/vectornet/data_tmp/qdraw_stitches/4518094242316288.svg'
# stroke_list = get_stroke_list(pm)