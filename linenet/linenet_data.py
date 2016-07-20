# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
import cairosvg
import matplotlib.pyplot as plt

import tensorflow as tf

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 48, # 48-24-12-6
                            """Image Size.""")
tf.app.flags.DEFINE_integer('xy_size', 8,
                            """# Coordinates of two lines.""")

SVG_TWO_LINES_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none" stroke="black" stroke-width="1">
    <path id="0" d="M {x1} {y1} L{x2} {y2}"/>
    <path id="1" d="M {x3} {y3} L{x4} {y4}"/>
</g></svg>"""

SVG_ONE_LINE_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none" stroke="black" stroke-width="1">
    <path id="0" d="M {x1} {y1} L{x2} {y2}"/>
</g></svg>"""

          
def batch():
    x_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    y_batch = np.empty([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1], dtype=np.float)
    for i in xrange(FLAGS.batch_size):
        xy = np.random.randint(low=0, high=FLAGS.image_size, size=FLAGS.xy_size)
        # print(xy.shape, xy.dtype)
        SVG_LINE1 = SVG_ONE_LINE_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size,
            x1=xy[0], y1=xy[1],
            x2=xy[2], y2=xy[3]
        )        
        SVG_TWO_LINES = SVG_TWO_LINES_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size,
            x1=xy[0], y1=xy[1],
            x2=xy[2], y2=xy[3],
            x3=xy[4], y3=xy[5],
            x4=xy[6], y4=xy[7]
        )

        # save y png
        y_name = os.path.join(FLAGS.data_dir, 'y_%d.png' % i)
        cairosvg.svg2png(bytestring=SVG_LINE1, write_to=y_name)

        # load and normalize y to [0, 1]
        y = imread(y_name)[:,:,3].astype(np.float) / 255.0
        y_batch[i, ...] = np.reshape(y, [FLAGS.image_size, FLAGS.image_size, 1])

        # plt.imshow(y, cmap=plt.cm.gray)
        # plt.show()

        # select a random point on line1
        line_ids = np.argwhere(y > 0.5)
        point_id = np.random.randint(len(line_ids))
        point_xy = line_ids[point_id]
        
        # save x png
        x_name = os.path.join(FLAGS.data_dir, 'x_%d.png' % i)
        cairosvg.svg2png(bytestring=SVG_TWO_LINES, write_to=x_name)

        
        # load and normalize y to [0, 0.1]
        x = imread(x_name)[:,:,3].astype(np.float) / (255.0 * 10.0)        
        x[point_xy[0], point_xy[1]] = 100.0
        
        # # debug
        # plt.imshow(x, cmap=plt.cm.gray)
        # plt.show()

        x_batch[i, ...] = np.reshape(x, [FLAGS.image_size, FLAGS.image_size, 1])
                    
    return x_batch, y_batch