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
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from LSTM_model_RUL import LstmRNN
# In[]
'''
命令行参数定义
'''

flags = tf.app.flags

flags.DEFINE_integer("input_size", 21, "Input size [21]")
flags.DEFINE_integer("output_size", 1, "Output size [1]")
flags.DEFINE_integer("num_steps", 30, "Num of steps [30]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("sample_size", 10, "Number of units to plot during training. [10]")

# In[]
FLAGS = flags.FLAGS
#打印命令行参数
pp = pprint.PrettyPrinter()
pp.pprint(FLAGS.__flags)
#print(FLAGS.__flags)

# In[]
'''
创建日志文件夹
'''
if not os.path.exists("logs"):
    os.mkdir("logs")
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

# In[]
train_X=dataset_RUL.train_X
train_y=-dataset_RUL.train_y
# In[]

# In[]
num_batches = int(len(train_X)) // batch_size#计算num_batches
if batch_size * num_batches < len(train_X):#避免由于整除舍去小数后无法完全覆盖所有样本
    num_batches += 1
    batch_indices = list(range(num_batches))
    random.shuffle(batch_indices)#将序列的所有元素随机排序
    for j in batch_indices:
        batch_X = train_X[j * batch_size: (j + 1) * batch_size]
        batch_y = train_y[j * batch_size: (j + 1) * batch_size]
# In[]
def plot_samples( sample_pred, sample_truth, image_path):
    figure=plt.figure()
    figure.set_figheight(5)
    figure.set_figwidth(8)
    plot_test, = plt.plot(sample_truth, label='real_RUL')
    plot_predicted, = plt.plot(sample_pred, label='predicted_RUL')
    plt.legend([plot_predicted, plot_test],['predicted', 'truth'])
    '''
    x_start=1000
    x_end=1060
    y_start=-1
    y_end=-0.2
    '''
    #plt.axis([x_start,x_end,y_start,y_end])
    plt.show()
    plt.savefig(image_path+'.png')
    plt.close()

# In[]
'''
# In[]
sample_size=10
test_list_indices=list(range(len(dataset_RUL.test_X_list)))
random.shuffle(test_list_indices)#随机打乱
sample_indices=test_list_indices[0:sample_size]
# In[]
test_X_list1=np.array(test_X_list)
test_y_list1=np.array(test_y_list)
# In[]
indice=0
image_path='hello'
sample_X=test_X_list[indice]
sample_y=test_y_list[indice]
sample_y_flattened=sample_y.reshape((-1,))
sample_pred=sample_y_flattened*1.1
# In[]
plot_samples( sample_pred, sample_truth, image_path)
# In[]
a=np.array([[[1,2],[3,4]],
            [[10,20],[30,40]],
            [[100,200],[300,400]]
        ])
b=np.array([[-1,-2],
            [-3,-4]
        ])
a[0,:,:]=b
c=a.reshape((-1,2))
d=c.reshape((3,2,2))
e=[1,2]
f=a[e]
# In[]
import numpy as np
a=np.array([
        [[1,-10],[2,-20],[3,-30]],
        [[2,-20],[3,-30],[4,-40]],
        [[3,-30],[4,-40],[5,-50]]
        ])
w=np.array([
        [[-100],[-200]],
        [[-200],[-300]],
        [[-300],[-400]]
        ])
b=np.array([
        [[1],[2],[3]],
        [[2],[3],[4]],
        [[3],[4],[5]]
        ])
c=np.matmul(a,w)+b
# In[]
w=np.array([[-100],[-200]])
b=np.array([[1],[2],[3]])
d=np.zeros((3,3,1))
for i in range(3):
    d[i]=np.matmul(a[i],w)+b
'''