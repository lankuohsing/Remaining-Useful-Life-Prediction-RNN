# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:05:57 2018

@author: lankuohsing
"""
# In[]
import tensorflow as tf
import pprint
import numpy as np
import random
import os
import time
from data_model_RUL import RULDataSet
import os
import pandas as pd
import tensorflow.contrib.slim as slim
from LSTM_model_RUL import LstmRNN
# In[]
'''
命令行参数定义
'''

flags = tf.app.flags

flags.DEFINE_integer("input_size", 12, "Input size [21]")
flags.DEFINE_integer("output_size", 1, "Output size [1]")
flags.DEFINE_integer("num_steps", 10, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 1, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("sample_size", 10, "Number of units to plot during training. [10]")
flags.DEFINE_string("logs_dir", "logs_97", "directory for logs. [logs]")
flags.DEFINE_string("plots_dir", "figures_97", "directory for plot figures. [figures]")
# In[]
FLAGS = flags.FLAGS
#打印命令行参数
pp = pprint.PrettyPrinter()
pp.pprint(tf.flags.FLAGS.__flags)

# In[]
'''
创建日志文件夹
'''
if not os.path.exists(FLAGS.logs_dir):
    os.mkdir(FLAGS.logs_dir)
# In[]
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
# In[]
batch_size=64
max_epoch=50
RUL_Data=RULDataSet()
dataset_RUL=RUL_Data

# In[]
'''
num_batches = int(len(dataset_RUL.train_X)) // batch_size#计算num_batches
if batch_size * num_batches < len(dataset_RUL.train_X):#避免由于整除舍去小数后无法完全覆盖所有样本
    num_batches += 1
    batch_indices = list(range(num_batches))
    random.shuffle(batch_indices)#将序列的所有元素随机排序
    for j in batch_indices:
        batch_X = dataset_RUL.train_X[j * batch_size: (j + 1) * batch_size]
        batch_y = dataset_RUL.train_y[j * batch_size: (j + 1) * batch_size]
'''
# In[]
train_X_list=dataset_RUL.train_X_list
train_y_list=dataset_RUL.train_y_list
# In[]
test_X_list=dataset_RUL.test_X_list
test_y_list=dataset_RUL.test_y_list
# In[]
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
# In[]
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
#print("run_config.batch_size:",run_config.batch_size)
with tf.Session(config=run_config) as sess:
    rnn_model = LstmRNN(
            sess,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            output_size=FLAGS.output_size,
            logs_dir=FLAGS.logs_dir,
            plots_dir=FLAGS.plots_dir,
            max_epoch=FLAGS.max_epoch

        )
    show_all_variables()
    RUL_Data=RULDataSet(
                 scaled_train_path='unit_number_RUL_97.csv',
                 scaled_test_path='test_FD001_scaled_selected.csv',
                 knee_point_path='knee_point_list.csv',
                 num_steps=FLAGS.num_steps,
                 test_ratio=0.1#测试集占数据集的比例
        )
    if FLAGS.train:
        rnn_model.train(RUL_Data, FLAGS)




