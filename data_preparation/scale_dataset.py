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
from get_knee import *
import numpy as np
import matplotlib.pyplot as plt
# In[]
sensor_index=[1, 2, 3, 6, 7, 10, 11, 12, 14, 16, 19, 20]
# In[]
path="train_FD001.xlsx"
feature_range=(0,1)#归一化的范围,类型为tuple
unit_number_list,raw_data,train_data_min_,train_data_max_=read_train_data_and_scale(path,
                                                    sensor_index)
# In[]
'''
test_path='test_FD001.xlsx'
import pandas as pd
#train_data_min_=np.reshape(train_data_min_,(1,len(train_data_min_)))
#train_data_max_=np.reshape(train_data_max_,(1,len(train_data_max_)))
test_DataFrame=pd.read_excel(test_path,sheet_name=0,header=None)#读取Excel表格中的数据
test_np=test_DataFrame.as_matrix()  #将test数据集放在一个NumPy array中
test_np_scaled=test_np
for i in range(0,21):
    if train_data_min_[i]==train_data_max_[i]:
        test_np_scaled[:,5+i]=0
    else:
        test_np_scaled[:,5+i]=(test_np[:,5+i]-train_data_min_[i])/(train_data_max_[i]-train_data_min_[i])
#test_np_scaled=(test_np[:,5:26]-train_data_min_)/(train_data_max_-train_data_min_)
#test_np_scaled=np.hstack((test_np[:,0:5],test_np_scaled))
#将归一化的数据保存在csv文件中
np.savetxt(test_path+'_scaled'+'.csv', test_np_scaled, delimiter = ',')
unit_number_redundant=test_np_scaled[:,0]  #提取出冗余的unit编号
unit_number=np.unique(unit_number_redundant)  #删除unit编号中的冗余部分
unit_nums=unit_number.shape[0]  #发动机编号数
#将每台发动机的运行数据存在一个二维列表中，将所有的二维列表存在一个list中
unit_number_list=[]
for i in range(0,unit_nums):
    condition_i=test_np_scaled[:,0]==i+1#找出对应编号的数据下标集合
    unit_index_i=np.where(condition_i)
    unit_number_i_index=unit_index_i[0]
    unit_number_i=test_np_scaled[unit_number_i_index,:]
    if(True):
        unit_number_i=unit_number_i[:,5:26]
    unit_number_list.append(unit_number_i)
'''
# In[]
test_path='test_FD001.xlsx'
test_unit_number_list,test_np=read_test_data_and_scale(test_path,
                                                       sensor_index,
                                                       True,train_data_min_,train_data_max_)
# In[]


# In[]
"""
#unit_number_scaled_0=unit_number_list_scaled[0]
for j in range(0,100):
    unit_number_scaled_0=unit_number_list[j]
    i=1
    #plt.figure(i)
    plt.plot(unit_number_scaled_0[:,i])
    #plt.axis([0, 250, 0, 50])
    #plt.show()
"""
"""
for i in range(0,unit_number_scaled_0.shape[1]):
    plt.figure(i)
    plt.plot(unit_number_scaled_0[:,i])
"""
