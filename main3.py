# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:59:39 2018
预测
@author: lankuohsing
"""

# In[]
import pandas as pd
import numpy as np
from read_save_data import read_unit_data
from generate_train import *
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
mpl.use('Agg')
from matplotlib import pyplot as plt
# In[]
path='unit_number_RUL_scaled.csv'
unit_number_RUL_scaled_list=read_unit_data(path)#读取归一化的传感器数据以及剩余寿命数据

# In[]
path2='knee_point_list.csv'
knee_point_DataFrame=pd.read_csv(path2,header=0,encoding='utf-8')
knee_point_np=knee_point_DataFrame.as_matrix()
# In[]
train_X_list,train_Y_list=generate_train_from_unit_list(unit_number_RUL_scaled_list,knee_point_np)

# In[]
import shutil
import os
#模型存储路径
MODEL_PATH="Models/model_FD001_1"
"""
if not os.path.exists(MODEL_PATH):  ###判断文件是否存在，返回布尔值
   os.makedirs(MODEL_PATH)
shutil.rmtree(MODEL_PATH)
"""
# In[]
"""
Hyperparameters
"""
learn = tf.contrib.learn
HIDDEN_SIZE = 30  # Lstm中隐藏节点的个数
NUM_LAYERS = 1  # LSTM的层数
TIMESTEPS = 10  # 循环神经网络的截断长度，也即input sequence的长度
TRAINING_STEPS = 100  # 训练轮数
BATCH_SIZE = 20  # batch大小
PREDICT_STEPS=10 #每一轮的预测点个数，也即output sequence长度
NUM_FEATURES=7#输入特征维数
# In[]
def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(num_units=HIDDEN_SIZE,forget_bias=1.0,state_is_tuple=True)
    return lstm_cell

# 定义lstm模型
def lstm_model(X, y):
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    print("X.shape:",X.shape)#(batch_size, 10, 21)
    print("y.shape:",y.shape)#(batch_size, 10, 1)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    print("outputs.shape:",outputs.shape)#(batch_size, 10, 30)
    #print("final_state.shape:",final_state[0].dtype)
    output = tf.reshape(outputs[:,:,:], [-1, HIDDEN_SIZE])
    print("output.shape:",output.shape)#(batch_size*10, 30)
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    #注意，这里不用在最后加一层softmax层，因为不是分类问题
    predictions = tf.contrib.layers.fully_connected(output, 1, None)
    print("predictions.shape:",predictions.shape)
    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    print("labels.shape:",labels.shape)#(batch_size*10,)
    print("predictions.shape:",predictions.shape)#(batch_size*10,)
    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                             optimizer="Adagrad",
                                             learning_rate=0.1)
    return predictions, loss, train_op
# In[]
# 进行训练
# 封装之前定义的lstm
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir=MODEL_PATH))
#regressor = learn.Estimator(model_fn=lstm_model, model_dir=MODEL_PATH)
# In[]
for i in range(len(train_X_list)):
    regressor.fit(train_X_list[i], train_Y_list[i], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# In[]
def final_data_for_plot(predicted_list,test_y):

    test_y_list=test_y.reshape(test_y.shape[0]*test_y.shape[1],1).tolist()

    final_predicted_list=[]
    final_test_y_list=[]
    for i in range(0,len(predicted_list)-PREDICT_STEPS+1):
        if i%(PREDICT_STEPS*PREDICT_STEPS)==0:
            final_predicted_list.extend(predicted_list[i:i+PREDICT_STEPS])
            final_test_y_list.extend(test_y_list[i:i+PREDICT_STEPS])
    final_predicted=np.array(final_predicted_list).reshape(len(final_predicted_list),1)
    final_test_y=np.array(final_test_y_list).reshape(len(final_test_y_list),1)
    return final_predicted, final_test_y


# In[]
for i in range(len(train_X_list)):
    regressor.score(train_X_list[i],train_Y_list[i])
    predicted_list_i = list(regressor.predict(train_X_list[i]))
    final_predicted_i, final_test_y_i=final_data_for_plot(predicted_list_i,train_Y_list[i])
    # 计算MSE
    rmse_i = np.sqrt(((final_predicted_i - final_test_y_i) ** 2).mean(axis=0))
    print("Mean Square Error is:%f" % rmse_i[0])
    # In[]
    figure1=plt.figure(i)
    figure1.set_figheight(5)
    figure1.set_figwidth(8)
    plot_test_i, = plt.plot(final_test_y_i, label='real_sin')
    plot_predicted_i, = plt.plot(final_predicted_i, label='predicted')
    plt.legend([plot_predicted_i, plot_test_i],['predicted', 'real_sin'])
    x_start=1000
    x_end=1060
    y_start=-1
    y_end=-0.2
    #plt.axis([x_start,x_end,y_start,y_end])
    plt.savefig('figures/test_'+'TIMESTEPS='+str(TIMESTEPS)+'PREDICT_STEPS='+str(i)+'.png')
    plt.show()