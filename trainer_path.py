from __future__ import print_function

import os
import numpy as np
from tqdm import trange

from models import *
from utils import save_image, convert_png2mp4

class Trainer(object):
    def __init__(self, config, batch_manager):
        tf.set_random_seed(config.random_seed)
        self.config = config
        self.batch_manager = batch_manager
        self.x, self.y = batch_manager.batch() # normalized input
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

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format
        if self.data_format == 'NCHW':
            self.x = nhwc_to_nchw(self.x)
            self.y = nhwc_to_nchw(self.y)

        self.start_step = config.start_step
        self.log_step = config.log_step
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
                self.x, self.conv_hidden_num, self.repeat_num, self.data_format)
        self.y_img = denorm_img(self.y_, self.data_format) # for debug

        self.build_test_model()
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

    def build_test_model(self):
        self.x_test = tf.placeholder(dtype=tf.float32, 
                                   shape=[self.b_num, self.height, self.width, 2])
        self.y_test_, _ = VDSR(
                self.x_test, self.conv_hidden_num, self.repeat_num, 'NHWC',
                train=False, reuse=True)
        self.y_test = denorm_img(self.y_test_, 'NHWC')

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

            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0 or step == self.max_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                loss = result['loss']
                assert not np.isnan(loss), 'Model diverged with loss = NaN'

                print("[{}/{}] Loss: {:.6f}".format(step, self.max_step, loss))

            if step % (self.log_step * 10) == 0 or step == self.max_step-1:
                self.generate(x_list, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.lr_update)

        # save last checkpoint..
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(self.sess, save_path, global_step=self.step)
        self.batch_manager.stop_thread()
    
    def test(self):
        # dirty way to bypass graph finilization error
        g = tf.get_default_graph()
        g._finalized = False

        self.build_test_model()

    def generate(self, x_samples, root_path=None, idx=None):
        generated = self.sess.run(self.y_test, {self.x_test: x_samples})
        y_path = os.path.join(root_path, 'y_{}.png'.format(idx))
        save_image(generated, y_path, nrow=self.b_num)
        print("[*] Samples saved: {}".format(y_path))