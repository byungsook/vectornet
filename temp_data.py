import os

import numpy as np
import scipy.misc
import scipy.stats

class DataSet(object):
    """ class for batch data 
    1-1. with a probability of 10%, use target image as input
    1-2. weighted random choice based on the image size
    2. extract 424x424 patch if the size of image is bigger than the patch size
    """
    def __init__(self, data_path, patch_h, patch_w):
        self._train_x_path = data_path + 'train/x/'
        self._train_y_path = data_path + 'train/y/'
        self._test_x_path = data_path + 'test/x/'
        self._test_y_path = data_path + 'test/y/'
        self._patch_h, self._patch_w = patch_h, patch_w
        # both x and y have files of the same name
        _, _, self._train_file_names = next(os.walk(self._train_x_path), (None, None, []))
        _, _, self._test_file_names = next(os.walk(self._test_x_path), (None, None, []))
        if self._test_file_names[0] == '.DS_Store':
            del self._test_file_names[0]
        # for training data
        self._img_size_list = []
        for train_file_name in self._train_file_names:
            train_file_name_no_ext, _ = os.path.splitext(train_file_name)
            img_size = train_file_name_no_ext.split('-')[2]
            self._img_size_list.append(int(img_size))

        mx_size, max_size = min(self._img_size_list), max(self._img_size_list)
        ds = float(max_size - mx_size)
        self._img_size_list = [(img_size-mx_size)/ds for img_size in self._img_size_list]
        sum_size = sum(self._img_size_list)
        self._prob = [img_size/sum_size for img_size in self._img_size_list]

        # for testing data
        test_size = len(self._test_file_names)
        self.test_x = np.empty([test_size, self._patch_h, self._patch_w]) 
        self.test_y = np.empty([test_size, self._patch_h, self._patch_w])
        
        for i in range(test_size):
            file_name = self._test_file_names[i]            
            x_file_name = self._test_x_path + file_name
            y_file_name = self._test_y_path + file_name
            x_img = scipy.misc.imread(x_file_name)
            y_img = scipy.misc.imread(y_file_name)
            self.test_x[i, ...] = scipy.stats.threshold(
                                    scipy.misc.imresize(x_img, (self._patch_h, self._patch_w)) / 255.0,
                                    threshmin=0.9, newval=0.0)
            self.test_y[i, ...] = scipy.misc.imresize(y_img, (self._patch_h, self._patch_w)) / 255.0

        self.test_x = np.reshape(self.test_x, [-1, self._patch_h, self._patch_w, 1]).astype(np.float32)
        self.test_y = np.reshape(self.test_y, [-1, self._patch_h, self._patch_w, 1]).astype(np.float32)


    def next_batch(self, batch_size):
        # make a batch for training
        batch_id = np.random.choice(len(self._train_file_names), batch_size, p=self._prob)
        use_target = np.random.choice([False, True], batch_size, p=[0.9, 0.1])

        batch_x = np.empty([batch_size, self._patch_h, self._patch_w])
        batch_y = np.empty([batch_size, self._patch_h, self._patch_w])
        for i in range(batch_size):
            file_name = self._train_file_names[batch_id[i]]
            y_file_name = self._train_y_path + file_name
            if use_target[i]:
                x_file_name = y_file_name
            else:
                x_file_name = self._train_x_path + file_name
            
            x_img = scipy.misc.imread(x_file_name)
            y_img = scipy.misc.imread(y_file_name)

            max_h0, max_w0 = x_img.shape[0] - self._patch_h + 1, x_img.shape[1] - self._patch_w + 1
            for _ in range(20):
                h0, w0 = np.random.randint(max_h0), np.random.randint(max_w0)
                batch_x[i, ...] = scipy.stats.threshold(x_img[h0:h0+self._patch_h, w0:w0+self._patch_w] / 255.0,
                                                        threshmin=0.9, newval=0.0)
                batch_y[i, ...] = y_img[h0:h0+self._patch_h, w0:w0+self._patch_w] / 255.0
                if np.amin(batch_x[i, ...]) == 1.0:
                    continue
                break

            # # check patch
            # scipy.misc.imshow(batch_x[i, ...])
            # scipy.misc.imshow(batch_y[i, ...])

        batch_x = np.reshape(batch_x, [-1, self._patch_h, self._patch_w, 1]).astype(np.float32)
        batch_y = np.reshape(batch_y, [-1, self._patch_h, self._patch_w, 1]).astype(np.float32)
        return batch_x, batch_y