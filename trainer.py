from __future__ import print_function

import os
import numpy as np
from tqdm import trange

from models import *
from utils import save_image

class Trainer(object):
    def __init__(self, config, batch_manager):
        tf.set_random_seed(config.random_seed)
        self.config = config
        self.batch_manager = batch_manager
        self.x, self.y = batch_manager.batch()
        self.xt = tf.placeholder(tf.float32, shape=int_shape(self.x))
        self.yt = tf.placeholder(tf.float32, shape=int_shape(self.y))
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.lr = tf.Variable(config.lr, name='lr')
        self.lr_update = tf.assign(self.lr, tf.maximum(self.lr*0.1, config.lr_lower_boundary), name='lr_update')

        self.height = config.height
        self.width = config.width
        self.b_num = config.batch_size
        self.conv_hidden_num = config.conv_hidden_num
        self.repeat_num = config.repeat_num
        self.use_l2 = config.use_l2
        self.use_norm = config.use_norm

        self.model_dir = config.model_dir

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format
        if self.data_format == 'NCHW':
            self.x = nhwc_to_nchw(self.x)
            self.y = nhwc_to_nchw(self.y)
            self.xt = nhwc_to_nchw(self.xt)
            self.yt = nhwc_to_nchw(self.yt)

        self.start_step = config.start_step
        self.log_step = config.log_step
        self.test_step = config.test_step
        self.max_step = config.max_step
        self.save_sec = config.save_sec
        self.lr_update_step = config.lr_update_step

        self.step = tf.Variable(self.start_step, name='step', trainable=False)

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=self.save_sec,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        if self.is_train:
            self.batch_manager.start_thread(self.sess)

    def build_model(self):
        self.y_, self.var = VDSR(
                self.x, self.conv_hidden_num, self.repeat_num, self.data_format, self.use_norm)
        self.y_img = denorm_img(self.y_, self.data_format) # for debug

        self.yt_, _ = VDSR(
                self.xt, self.conv_hidden_num, self.repeat_num, self.data_format, self.use_norm,
                train=False, reuse=True)
        self.yt_ = tf.clip_by_value(self.yt_, 0, 1)
        self.yt_img = denorm_img(self.yt_, self.data_format)

        show_all_variables()        

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(self.config.optimizer))

        optimizer = optimizer(self.lr, beta1=self.beta1, beta2=self.beta2)

        # losses
        # l1 and l2
        self.loss_l1 = tf.reduce_mean(tf.abs(self.y_ - self.y))
        self.loss_l2 = tf.reduce_mean(tf.squared_difference(self.y_, self.y))

        # total
        if self.use_l2:
            self.loss = self.loss_l2
        else:
            self.loss = self.loss_l1

        # test loss
        self.tl1 = 1 - tf.reduce_mean(tf.abs(self.yt_ - self.yt))
        self.tl2 = 1 - tf.reduce_mean(tf.squared_difference(self.yt_, self.yt))
        self.test_acc_l1 = tf.placeholder(tf.float32)
        self.test_acc_l2 = tf.placeholder(tf.float32)
        self.test_acc_iou = tf.placeholder(tf.float32)

        self.optim = optimizer.minimize(self.loss, global_step=self.step, var_list=self.var)
 
        summary = [
            tf.summary.image("y", self.y_img),

            tf.summary.scalar("loss/loss", self.loss),
            tf.summary.scalar("loss/loss_l1", self.loss_l1),
            tf.summary.scalar("loss/loss_l2", self.loss_l2),
           
            tf.summary.scalar("misc/lr", self.lr),
            tf.summary.scalar('misc/q', self.batch_manager.q.size())
        ]

        self.summary_op = tf.summary.merge(summary)

        summary = [
            tf.summary.image("x_sample", denorm_img(self.x, self.data_format)),
            tf.summary.image("y_sample", denorm_img(self.y, self.data_format)),
        ]

        self.summary_once = tf.summary.merge(summary) # call just once

        summary = [
            tf.summary.scalar("loss/test_acc_l1", self.test_acc_l1),
            tf.summary.scalar("loss/test_acc_l2", self.test_acc_l2),
            tf.summary.scalar("loss/test_acc_iou", self.test_acc_iou),
        ]

        self.summary_test = tf.summary.merge(summary)

    def train(self):
        x_list, xs, ys, sample_list = self.batch_manager.random_list(self.b_num)
        save_image(xs, '{}/x_gt.png'.format(self.model_dir))
        save_image(ys, '{}/y_gt.png'.format(self.model_dir))

        with open('{}/gt.txt'.format(self.model_dir), 'w') as f:
            for sample in sample_list:
                f.write(sample + '\n')
        
        # call once
        summary_once = self.sess.run(self.summary_once)
        self.summary_writer.add_summary(summary_once, 0)
        self.summary_writer.flush()

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "optim": self.optim,
                "loss": self.loss,
            }           

            if step % self.log_step == 0 or step == self.max_step-1:
                fetch_dict.update({
                    "summary": self.summary_op,                    
                })

            if step % self.test_step == self.test_step-1 or step == self.max_step-1:
                l1, l2, iou, nb = 0, 0, 0, 0
                for x, y in self.batch_manager.test_batch():
                    if self.data_format == 'NCHW':
                        x = to_nchw_numpy(x)
                        y = to_nchw_numpy(y)
                    tl1, tl2, y_ = self.sess.run([self.tl1, self.tl2, self.yt_], {self.xt: x, self.yt: y})
                    l1 += tl1
                    l2 += tl2
                    nb += 1

                    # iou
                    y_I = np.logical_and(y>0, y_>0)
                    y_I_sum = np.sum(y_I, axis=(1, 2, 3))
                    y_U = np.logical_or(y>0, y_>0)
                    y_U_sum = np.sum(y_U, axis=(1, 2, 3))
                    # print(y_I_sum, y_U_sum)
                    nonzero_id = np.where(y_U_sum != 0)[0]
                    if nonzero_id.shape[0] == 0:
                        acc = 1.0
                    else:
                        acc = np.average(y_I_sum[nonzero_id] / y_U_sum[nonzero_id])
                    iou += acc

                    if nb > 500:
                        break

                l1 /= float(nb)
                l2 /= float(nb)
                iou /= float(nb)
                    
                summary_test = self.sess.run(self.summary_test, 
                              {self.test_acc_l1: l1, self.test_acc_l2: l2, self.test_acc_iou: iou})
                self.summary_writer.add_summary(summary_test, step)
                self.summary_writer.flush()

            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0 or step == self.max_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                loss = result['loss']
                assert not np.isnan(loss), 'Model diverged with loss = NaN'

                print("\n[{}/{}] Loss: {:.6f}".format(step, self.max_step, loss))

            if step % (self.log_step * 10) == 0 or step == self.max_step-1:
                self.generate(x_list, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()

    def generate(self, x_samples, root_path=None, idx=None):
        if self.data_format == 'NCHW':
            x_samples = to_nchw_numpy(x_samples)
        generated = self.sess.run(self.yt_img, {self.xt: x_samples})
        y_path = os.path.join(root_path, 'y_{}.png'.format(idx))
        save_image(generated, y_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(y_path))