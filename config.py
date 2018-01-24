#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--width', type=int, default=64)
net_arg.add_argument('--height', type=int, default=64)
net_arg.add_argument('--conv_hidden_num', type=int, default=64,
                     choices=[64, 128, 256])
net_arg.add_argument('--repeat_num', type=int, default=20,
                     choices=[16, 20, 32])
net_arg.add_argument('--use_l2', type=str2bool, default=True)
net_arg.add_argument('--use_norm', type=str2bool, default=True)
net_arg.add_argument('--archi', type=str, default='path',
                     choices=['path','overlap'])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='data')
data_arg.add_argument('--dataset', type=str, default='line',
                      choices=['line','ch','kanji','baseball','cat'])
data_arg.add_argument('--batch_size', type=int, default=8)
data_arg.add_argument('--num_worker', type=int, default=16)
# line
data_arg.add_argument('--num_strokes', type=int, default=4)
data_arg.add_argument('--stroke_type', type=int, default=2)
data_arg.add_argument('--min_length', type=int, default=10)
data_arg.add_argument('--max_stroke_width', type=int, default=2) # 4 for varying w.

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--gpu_id', type=str, default='0')
train_arg.add_argument('--start_step', type=int, default=0)
train_arg.add_argument('--max_step', type=int, default=50000) # 2000
train_arg.add_argument('--lr_update_step', type=int, default=20000)
train_arg.add_argument('--lr', type=float, default=0.005)
train_arg.add_argument('--lr_lower_boundary', type=float, default=0.00001)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)

# vectorize
vect_arg = add_argument_group('Vectorize')
vect_arg.add_argument('--load_pathnet', type=str, default='')
vect_arg.add_argument('--load_overlapnet', type=str, default='')
vect_arg.add_argument('--num_test', type=int, default=100)
vect_arg.add_argument('--max_label', type=int, default=128)
vect_arg.add_argument('--label_cost', type=int, default=0)
vect_arg.add_argument('--sigma_neighbor', type=float, default=8.0)
vect_arg.add_argument('--sigma_predict', type=float, default=0.7)
vect_arg.add_argument('--neighbor_sample', type=float, default=1)
vect_arg.add_argument('--find_overlap', type=str2bool, default=True)
vect_arg.add_argument('--overlap_threshold', type=float, default=0.5)
vect_arg.add_argument('--test_batch_size', type=int, default=512)
vect_arg.add_argument('--mp', type=str2bool, default=True)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--test_step', type=int, default=10000) # 1000
misc_arg.add_argument('--save_sec', type=int, default=900)
misc_arg.add_argument('--log_dir', type=str, default='log')
misc_arg.add_argument('--tag', type=str, default='test')
misc_arg.add_argument('--random_seed', type=int, default=123)


def get_config():
    config, unparsed = parser.parse_known_args()
    
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id # "0, 1" for multiple

    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    # data_format = 'NHWC' # for debug
    setattr(config, 'data_format', data_format)
    return config, unparsed
