# Copyright (c) 2016 Byungsoo Kim. All Rights Reserved.
# 
# Byungsoo Kim, ETH Zurich
# kimby@student.ethz.ch, http://byungsoo.me
# ==============================================================================

"""Routine for decoding the Cubic Bezier Curve binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from scipy.misc import imread
import cairosvg
import matplotlib.pyplot as plt
import tarfile
import shutil

import tensorflow as tf

# parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data',
                            """Path to the Beziernet data directory.""")

tf.app.flags.DEFINE_integer('image_size', 96, # 96-48-24-12-6
                            """Image Size.""")
tf.app.flags.DEFINE_integer('xy_size', 8,
                            """# Coordinates of Bezier Curve.""")
tf.app.flags.DEFINE_integer('num_examples', 100000,
                            """# examples.""")
tf.app.flags.DEFINE_integer('num_examples_per_bin', 10000,
                            """# examples per bin.""")
tf.app.flags.DEFINE_integer('num_bins', 10, #int(FLAGS.num_examples / FLAGS.num_examples_per_bin),
                            """# bins.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', FLAGS.num_examples_per_bin * (FLAGS.num_bins-1),
                            """# examples per epoch for training.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_test', FLAGS.num_examples_per_bin,
                            """# examples per epoch for testing.""")


SVG_DIR = os.path.join(FLAGS.data_dir, 'svg')
PNG_DIR = os.path.join(FLAGS.data_dir, 'png')
LABEL_DIR = os.path.join(FLAGS.data_dir, 'label')
BIN_DIR = os.path.join(FLAGS.data_dir, 'bin')
TAR_DIR = os.path.join(FLAGS.data_dir, 'tar')
TAR_BIN_FILE_NAME = os.path.join(TAR_DIR, 'svg_bin.tar.gz')

SVG_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
<g fill="none" stroke="black" stroke-width="1">
    <path id="0" d="M {sx} {sy} C {cx1} {cy1} {cx2} {cy2} {tx} {ty}"/>
</g></svg>"""


def _generate_image_and_label_batch(image, xy, min_queue_examples, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 2-D Tensor of [height, width] of type.float32.
        xy: 1-D Tensor of [FLAGS.xy_size] type.uint8
        min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 1] size.
        xys: Coordinates. 2D tensor of [batch_size, FLAGS.xy_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, xys = tf.train.shuffle_batch(
            [image, xy],
            batch_size=FLAGS.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * FLAGS.batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, xys = tf.train.batch(
            [image, xy],
            batch_size=FLAGS.batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * FLAGS.batch_size)

    # # Display the training images in the visualizer.
    # tf.image_summary('images', images, max_images=FLAGS.max_images)

    return images, tf.reshape(xys, [FLAGS.batch_size, FLAGS.xy_size])


def _read_bezier_bin(filename_queue):
    """Reads and parses examples from Bezier bin data files.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result (FLAGS.image_size)
            width: number of columns in the result (FLAGS.image_size)
            key: a scalar string Tensor describing the filename & record number
            for this example.
            xy: an [xy_dim] uint8 Tensor with the coordinates of a bezier curve
            uint8image: a [height, width] uint8 Tensor with the image data
    """
    class BezierRecord(object):
        pass
    result = BezierRecord()

    # Dimensions of the images in the Bezier dataset.
    result.xy_dim = FLAGS.xy_size
    xy_bytes = result.xy_dim
    result.height = FLAGS.image_size
    result.width = FLAGS.image_size
    image_bytes = result.height * result.width
    record_bytes = xy_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.xy = tf.slice(record_bytes, [0], [xy_bytes])

    # The remaining bytes after the xy represent the image, which we reshape
    # from [height * width] to [height, width, 1].
    result.uint8image = tf.reshape(tf.slice(record_bytes, [xy_bytes], [image_bytes]),
                           [result.height, result.width, 1])

    return result


def svg_to_png(xy):
    np.clip(xy, a_min=1, a_max=FLAGS.image_size, out=xy)
    xy = xy.astype(np.int)
    png_img = np.empty([FLAGS.max_images, FLAGS.image_size, FLAGS.image_size], dtype=np.uint8)
    for i in xrange(FLAGS.max_images):
        SVG = SVG_TEMPLATE.format(
                width=FLAGS.image_size,
                height=FLAGS.image_size,
                sx=xy[i, 0], sy=xy[i, 1],
                cx1=xy[i, 2], cy1=xy[i, 3],
                cx2=xy[i, 4], cy2=xy[i, 5],
                tx=xy[i, 6], ty=xy[i, 7]
            )

        # save png
        png_file_name = 'tmp.png'
        cairosvg.svg2png(bytestring=SVG, write_to=png_file_name)

        png_img[i, ...] = imread(png_file_name)[:,:,3] 
        os.remove(png_file_name)
    
    return np.reshape(png_img, [-1, FLAGS.image_size, FLAGS.image_size, 1])


def inputs(is_train=True):
    """Construct input for Beziernet evaluation using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, FLAGS.image_size, FLAGS.image_size, 1] size.
        xys: Coordinates of bezier curves. 2D tensor of [batch_size, FLAGS.xy_size] size.
    """
    if is_train:
        filenames = [os.path.join(BIN_DIR, '%d.bin' % i) for i in xrange(1, FLAGS.num_bins)]
        num_examples_per_epoch = FLAGS.num_examples_per_epoch_for_train
    else:
        filenames = [os.path.join(BIN_DIR, 'test.bin')]
        num_examples_per_epoch = FLAGS.num_examples_per_epoch_for_test

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = _read_bezier_bin(filename_queue)
    normalized_image = tf.image.convert_image_dtype(read_input.uint8image, tf.float32)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                            min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(normalized_image, read_input.xy,
                                         min_queue_examples, shuffle=False)


def extract_bezier_bin():
    tarfile.open(TAR_BIN_FILE_NAME, 'r:gz').extractall()


def generate_bezier_bin():    
    if not os.path.exists(SVG_DIR):
        os.makedirs(SVG_DIR)
    
    if not os.path.exists(PNG_DIR):
        os.makedirs(PNG_DIR)
        
    label_file_name =  os.path.join(LABEL_DIR, 'label.txt')
    if not os.path.exists(LABEL_DIR):
        os.makedirs(LABEL_DIR)
    elif os.path.exists(label_file_name):
        os.remove(label_file_name)
    label_f = open(label_file_name, 'a')

    if not os.path.exists(BIN_DIR):
        os.makedirs(BIN_DIR)
    bin_id = 1
    bin_file_name = os.path.join(BIN_DIR, '%d.bin' % bin_id)
    bin_f = open(bin_file_name, 'w')

    print('create cubic bezier curves')
    np.random.seed(0)
    for i in xrange(1, FLAGS.num_examples+1):
        # create a cubic bezier
        xy = np.random.randint(low=1, high=FLAGS.image_size, size=FLAGS.xy_size)
        # print(xy.shape, xy.dtype)
        SVG = SVG_TEMPLATE.format(
            width=FLAGS.image_size,
            height=FLAGS.image_size,
            sx=xy[0], sy=xy[1],
            cx1=xy[2], cy1=xy[3],
            cx2=xy[4], cy2=xy[5],
            tx=xy[6], ty=xy[7]
        )

        # save svg
        svg_file_name = os.path.join(SVG_DIR, '%d.svg' % i)
        with open(svg_file_name, 'w') as f:
            f.write(SVG)
        
        # save png
        png_file_name = os.path.join(PNG_DIR, '%d.png' % i)
        cairosvg.svg2png(bytestring=SVG, write_to=png_file_name)
        
        # save label
        for l in np.nditer(xy):            
            label_f.write('%d ' % l)
        label_f.write('\n')

        # save binary
        png_img = imread(png_file_name)[:,:,3]
        # print(png_img.shape, png_img.dtype)
        # plt.imshow(png_img, cmap=plt.cm.gray)
        # plt.show()

        # !! only if FLAGS.image_size < 255, otherwise use uint16
        bin_f.write(xy.astype(np.uint8).tobytes())  
        bin_f.write(png_img.tobytes())
        if i % FLAGS.num_examples_per_bin == 0:
            bin_f.close()

            # # test binary reading
            # filename_queue = tf.train.string_input_producer([bin_file_name])            
            # read_input = _read_bezier_bin(filename_queue)
            # sess = tf.InteractiveSession()
            # tf.train.start_queue_runners(sess=sess)            
            # img_eval, xys_eval = sess.run([read_input.uint8image, read_input.xy])
            # xys_eval = svg_to_png(np.reshape(xys_eval, [1, 8]))
            # plt.imshow(np.reshape(img_eval, [FLAGS.image_size, FLAGS.image_size]), cmap=plt.cm.gray)
            # plt.show()
            # plt.imshow(np.reshape(xys_eval, [FLAGS.image_size, FLAGS.image_size]), cmap=plt.cm.gray)
            # plt.show() 

            bin_id += 1
            if bin_id < FLAGS.num_bins:
                bin_file_name = os.path.join(BIN_DIR, '%d.bin' % bin_id)
            else:
                bin_file_name = os.path.join(BIN_DIR, 'test.bin')
            
            if i < FLAGS.num_examples:
                bin_f = open(bin_file_name, 'w')

    label_f.close()
    
    if not os.path.exists(TAR_DIR):
        os.makedirs(TAR_DIR)
    
    print('tar all bin files')
    with tarfile.open(TAR_BIN_FILE_NAME, 'w:gz') as tar:
        tar.add(BIN_DIR)

    print('tar and remove all other files')
    tar_other_file_name = os.path.join(TAR_DIR, 'svg_others.tar.gz')
    with tarfile.open(tar_other_file_name, 'w:gz') as tar:
        for name in [SVG_DIR, PNG_DIR, LABEL_DIR]:
            tar.add(name)
            shutil.rmtree(name)

    print('done')
