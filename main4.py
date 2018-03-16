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
flags.DEFINE_string("run_mode", "train", "runing mode,train or test. [train]")
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
flags.DEFINE_integer("max_epoch", 1, "Total training epoches. [50]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("sample_size", 10, "Number of units to plot during training. [10]")
flags.DEFINE_string("logs_dir", "logs_97_2", "directory for logs. [logs]")
flags.DEFINE_string("plots_dir", "figures_97_2", "directory for plot figures. [figures]")
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
final_test_RUL=np.array([112,98,69,82,91,93,91,95,111,96,97,124,95,
                         107,83,84,50,28,87,16,57,111,113,20,145,119,
                         66,97,90,115,8,48,106,7,11,19,21,50,142,28,
                         18,10,59,109,114,47,135,92,21,79,114,29,26,
                         97,137,15,103,37,114,100,21,54,72,28,128,14,
                         77,8,121,94,118,50,131,126,113,10,34,107,63,
                         90,8,9,137,58,118,89,116,115,136,28,38,20,85,
                         55,128,137,82,59,117,20])
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
S_list=[]
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
# In[]
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
#print("run_config.batch_size:",run_config.batch_size)
for FLAGS.num_layers in [1,2,3,4]:
    for FLAGS.lstm_size in [100,110,120,130,140,150,160]:
        for FLAGS.num_steps in [5,10,15,20,25,30]:
            tf.reset_default_graph()
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
                if FLAGS.run_mode=="train":
                    rnn_model.train(RUL_Data, FLAGS)
                    '''
                    final_test_pred_list=rnn_model.test(RUL_Data, FLAGS)
                    final_test_pred_last_np=np.array([final_test_pred_list[i][0][-1] for i in range(len(final_test_pred_list))])
                    a0=final_test_pred_last_np - final_test_RUL
                    a=np.sign(a0)*a0/(11.5-1.5*np.sign(a0))
                    b=np.exp(a)-1
                    S=np.sum(b)
                    print("S:",S)
                    S_list.append(S)
                    '''
                else:
                    rnn_model.load()
                    final_test_pred_list=rnn_model.test(RUL_Data, FLAGS)
        # In[]



# In[]
file=open('S_list.txt','w')
file.write("S_list:"+str(S_list)+"\n");
file.close()
