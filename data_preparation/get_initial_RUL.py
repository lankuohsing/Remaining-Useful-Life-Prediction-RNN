# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:42:22 2018
读取原始的发动机运行数据，归一化并进行特征选择后后存在‘train_FD001_scaled_to_(0, 1)_selected.csv’中，
寻找健康指数衰退拐点，画图保存，并将拐点数据存在‘knee_point_list.csv’中
读取测试集并按照训练集一样的放缩尺度进行放缩，保存在‘test_FD001_scaled_selected.csv'中'
@author: lankuohsing
"""

# In[]
"""
目的是将自定义的模块的路径加入到python的搜索路径目录中
"""
import sys
import os
print(os.getcwd())#显示当前工作目录
path='D:\Projects\Github\Remaining-Useful-Life-Prediction-RNN\data_preparation'
sys.path.append(path)
print(sys.path)
os.chdir(path)#修改当前工作目录
print(os.getcwd())#显示当前工作目录
# In[]
from read_save_data import read_train_data_and_scale
from read_save_data import read_test_data_and_scale
from read_save_data import read_scaled_train_data
from get_knee import *
import numpy as np
import matplotlib.pyplot as plt
# In[]
sensor_index=[1, 2, 3, 6, 7, 10, 11, 12, 14, 16, 19, 20]
sensor_decrease=[6, 11, 19, 20]
sensor_increase=[1, 2, 3, 7, 10, 12, 14, 16]
# In[]
sensor_decrease_in_selected=[sensor_index.index(i) for i in sensor_decrease]
sensor_increase_in_selected=[sensor_index.index(i) for i in sensor_increase]

# In[]
train_scaled_path='train_FD001_scaled_to_(0, 1)_selected.csv'
train_scaled_sensor_list=read_scaled_train_data(train_scaled_path,isCut=True)
# In[]
train_scaled_decrease_sensor_list=[]
train_scaled_increase_sensor_list=[]
for i in range(len(train_scaled_sensor_list)):
    train_scaled_decrease_sensor_list_i=train_scaled_sensor_list[i][:,sensor_decrease_in_selected]
    train_scaled_decrease_sensor_list.append(train_scaled_decrease_sensor_list_i)
    train_scaled_increase_sensor_list_i=train_scaled_sensor_list[i][:,sensor_increase_in_selected]
    train_scaled_increase_sensor_list.append(train_scaled_increase_sensor_list_i)

# In[]
increase_knee_point_list=[]
window=10
dot_threshold=0.05
between_cluster_se_dot_list_list=[]
for i in range(0,len(train_scaled_increase_sensor_list)):
    train_scaled_decrease_sensor_list_i=train_scaled_increase_sensor_list[i]
    between_cluster_se_list_i,between_cluster_se_dot_list_i=\
    get_between_cluster_se(train_scaled_decrease_sensor_list_i,window=window)
    between_cluster_se_dot_list_list.append(between_cluster_se_dot_list_i)
    '''
    plt.figure(i*2)
    plot_between_cluster_se=plt.plot(between_cluster_se_list_i,'r.')
    plt.savefig('plot_distance1/'+str(i)+'th unit\'s between_cluster_se_list'+'.png')
    plt.figure(i*2+1)
    plot_between_cluster_se_dot=plt.plot(between_cluster_se_dot_list_i,'b.')
    plt.savefig('plot_distance1/'+str(i)+'th unit\'s between_cluster_se_dot_list'+'.png')
    '''
    increase_knee_point_i=\
    get_knee_point(between_cluster_se_list_i,
                   between_cluster_se_dot_list_i,
                   dot_threshold=dot_threshold,
                   window=window)
    increase_knee_point_list.append(increase_knee_point_i)
# In[]
mean_knee_point_increase=np.mean(np.array(increase_knee_point_list))

file=open('increase_knee_point_list.txt','w')
file.write("increase_knee_point_list:"+str(increase_knee_point_list)+"\n");
file.write("mean_knee_point_increase:"+str(mean_knee_point_increase)+"\n");
file.close()
#se_between_0=
# In[]
decrease_knee_point_list=[]
window=10
dot_threshold=0.05
between_cluster_se_dot_list_list=[]
for i in range(0,len(train_scaled_decrease_sensor_list)):
    train_scaled_decrease_sensor_list_i=train_scaled_decrease_sensor_list[i]
    between_cluster_se_list_i,between_cluster_se_dot_list_i=\
    get_between_cluster_se(train_scaled_decrease_sensor_list_i,window=window)
    between_cluster_se_dot_list_list.append(between_cluster_se_dot_list_i)
    '''
    plt.figure(i*2)
    plot_between_cluster_se=plt.plot(between_cluster_se_list_i,'r.')
    plt.savefig('plot_distance1/'+str(i)+'th unit\'s between_cluster_se_list'+'.png')
    plt.figure(i*2+1)
    plot_between_cluster_se_dot=plt.plot(between_cluster_se_dot_list_i,'b.')
    plt.savefig('plot_distance1/'+str(i)+'th unit\'s between_cluster_se_dot_list'+'.png')
    '''
    decrease_knee_point_i=\
    get_knee_point(between_cluster_se_list_i,
                   between_cluster_se_dot_list_i,
                   dot_threshold=dot_threshold,
                   window=window)
    decrease_knee_point_list.append(decrease_knee_point_i)
# In[]
mean_knee_point_decrease=np.mean(np.array(decrease_knee_point_list))

file=open('decrease_knee_point_list.txt','w')
file.write("decrease_knee_point_list:"+str(decrease_knee_point_list)+"\n");
file.write("mean_knee_point_decrease:"+str(mean_knee_point_decrease)+"\n");
file.close()
