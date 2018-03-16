# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:26:31 2018
功能：将计算好的RUL连同归一化的传感器数据以及没有归一化的操作模式数据存到‘unit_number_RUL.csv’文件中
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
from read_save_data import read_scaled_train_data
import numpy as np
import pandas as pd
# In[]
path='train_FD001_scaled_to_(0, 1)_selected.csv'
unit_number_list=read_scaled_train_data(path,isCut=False)
path2='knee_point_list.csv'
knee_point_DataFrame=pd.read_csv(path2,header=0,encoding='utf-8')
knee_point_np=knee_point_DataFrame.as_matrix()
# In[]

window=10
unit_number_list_i=unit_number_list[0]#这是一个二维数组
max_life=unit_number_list_i.shape[0]-(knee_point_np[0,0]+window)
unit_RUL_i=np.ones((unit_number_list_i.shape[0],1))*max_life
knee_index=knee_point_np[0,0]+window-1
unit_RUL_i[knee_index+1:unit_RUL_i.shape[0],0]=range(max_life-1,-1,-1)

# In[]
window=10
for i in range(0,len(unit_number_list)):
    unit_number_list_i=unit_number_list[i]#这是一个二维数组
    #max_life=unit_number_list_i.shape[0]-(knee_point_np[i,0]+window)
    max_life=97
    unit_RUL_i=np.ones((unit_number_list_i.shape[0],1))*max_life
    knee_index=knee_point_np[i,0]+window-1
    unit_RUL_i[unit_RUL_i.shape[0]-max_life:unit_RUL_i.shape[0],0]=range(max_life-1,-1,-1)
    unit_number_list[i]=np.hstack((unit_number_list[i],unit_RUL_i))
# In[]
unit_number_RUL=unit_number_list[0]
for i in range(1,len(unit_number_list)):
    unit_number_RUL=np.vstack((unit_number_RUL,unit_number_list[i]))
# In[]
np.savetxt('unit_number_RUL_97.csv', unit_number_RUL, delimiter = ',')
# In[]
from sklearn.preprocessing import MinMaxScaler
feature_range=(0,1)
unit_number_RUL_scaled_list=[]
for i in range(0,len(unit_number_list)):
    unit_number_RUL_scaled_i=unit_number_list[i]
    scaler = MinMaxScaler(copy=True,feature_range=feature_range)#copy=True保留原始数据矩阵
    unit_number_RUL_scaled_i[:,-1]=scaler.fit_transform(
            unit_number_RUL_scaled_i[:,-1].\
            reshape((unit_number_RUL_scaled_i.shape[0],1))).flatten()
    unit_number_RUL_scaled_list.append(unit_number_RUL_scaled_i)
# In[]
unit_number_RUL_scaled=unit_number_RUL_scaled_list[0]
for i in range(1,len(unit_number_list)):
    unit_number_RUL_scaled=np.vstack((unit_number_RUL_scaled,unit_number_RUL_scaled_list[i]))
# In[]
np.savetxt('unit_number_RUL_97_scaled.csv', unit_number_RUL_scaled, delimiter = ',')
# In[]
# In[]
scaled_test_path='test_FD001_scaled_selected.csv'
test_unit_number_list=read_scaled_train_data(scaled_test_path,isCut=True)
# In[]
f=open('RUL_FD001.txt')
raw_txt=f.read()
print(raw_txt)
f.close()
str_list=raw_txt.split('\n')
str_list_last=str_list[100]
if(str_list_last==''):
    print("bingo")
test_RUL_list=[]
for i in range(len(str_list)):
    if (str_list[i]!=''):
        test_RUL_list.append(int(str_list[i]))
# In[]
import os,sys


#查看当前工作目录
print("当前的工作目录为：%s" %os.getcwd())