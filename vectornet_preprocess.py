"""
(paper) 68 pairs, average of 1280.0x1662.7, min 630x630, max 2416x3219
1. data augmentation (tone, slur, noise) -> 4 times
2. downscale to 7/6 ~ 14/6 -> 9 times
4. randomly rotate [-180 180] and flip horizontally
3. (only input) normalize to [0,1] and threshold (<0.9 -> 0, outputs have similar tones)
"""

import os

import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.stats

DATA_PATH = 'data/train/org/'
PREPROCESSED_X_PATH = 'data/train/x/'
PREPROCESSED_Y_PATH = 'data/train/y/'
PATCH_H, PATCH_W = 424, 424

current_path = os.getcwd()
# if release mode
if not current_path.endswith('vectornet'):
    working_path = current_path + '/vectornet'
    os.chdir(working_path)

np.random.seed(0)

_, _, file_names = next(os.walk(DATA_PATH), (None, None, []))

down_ratios = [6.0 / r for r in range(7, 15)]
down_ratios.insert(0, 1.0)

for file_name in file_names:
    img_file_name = DATA_PATH + file_name
    file_name_no_ext, _ = os.path.splitext(file_name)

    x_img = scipy.misc.imread(img_file_name)
    y_img = x_img

    ### TEMP !!!
    random_resize = 1.5 * np.random.rand() + 0.5
    x_img = scipy.misc.imresize(x_img, random_resize)
    y_img = scipy.misc.imresize(y_img, random_resize)
    ### TEMP !!!

    for r in down_ratios:
        if r < 1.0:
            transformed_x_img = scipy.misc.imresize(x_img, r)
            transformed_y_img = scipy.misc.imresize(y_img, r)
            if (transformed_x_img.shape[0] < PATCH_H or
                    transformed_x_img.shape[1] < PATCH_W):
                continue
        else:
            transformed_x_img = x_img
            transformed_y_img = y_img
        rotation_degree = 360.0 * np.random.random() - 180.0
        transformed_x_img = scipy.ndimage.rotate(transformed_x_img, rotation_degree, cval=255)
        transformed_y_img = scipy.ndimage.rotate(transformed_y_img, rotation_degree, cval=255)
        if np.random.randint(2) == 0:
            transformed_x_img = np.flipud(transformed_x_img)
            transformed_y_img = np.flipud(transformed_y_img)
        transformed_x_img = scipy.stats.threshold(transformed_x_img.astype(np.float32) / 255.0, threshmin=0.9, newval=0.0)

        # save to image
        new_size = transformed_x_img.shape[0] * transformed_x_img.shape[1]
        new_file_name = file_name_no_ext + "-%1.2f-%d.png" % (r, new_size)
        preprocessed_x_file_name = PREPROCESSED_X_PATH + new_file_name
        preprocessed_y_file_name = PREPROCESSED_Y_PATH + new_file_name
        scipy.misc.imsave(preprocessed_x_file_name, transformed_x_img)
        scipy.misc.imsave(preprocessed_y_file_name, transformed_y_img)
