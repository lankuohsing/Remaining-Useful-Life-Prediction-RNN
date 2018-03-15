# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:42:22 2018
读取原始的发动机运行数据，进行特征选择
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
from read_save_data import read_unit_data
from get_knee import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# In[]
train_path="train_FD001.xlsx"
'''
path=train_path
if 1>0:
    if path.split('.')[-1]=="xlsx":
        print("read xlsx......")
        unit_DataFrame=pd.read_excel(path,sheet_name=0,header=None)#读取Excel表格中的数据
    else:
        print("read csv......")
        unit_DataFrame=pd.read_csv(path,header=None,encoding='utf-8')
'''
train_unit_number_list=read_unit_data(train_path,isCut=True)
# In[]
train_unit_num=len(train_unit_number_list)
sensor_num=train_unit_number_list[0].shape[1]#传感器个数
# In[]
train_RUL_list=[]#所有单元发动机的RUL
for i in range(0,train_unit_num):
    train_RUL_list_i=np.array(range(train_unit_number_list[i].shape[0]-1,-1,-1))
    train_RUL_list.append(train_RUL_list_i)
# In[]
for i in range(sensor_num):
    plt.figure(i,figsize=(8,6))
    for j in range(train_unit_num):
        plt.plot(train_unit_number_list[j][:,i],'b')
    plt.tick_params(labelsize=20)
    plt.savefig('plot_sensor/'+str(i)+'_sensor'+'.png')
    plt.show()
# In[]
from scipy.stats import pearsonr
pearson_coef=np.zeros((train_unit_num,sensor_num))
for i in range(sensor_num):
    for j in range(train_unit_num):
        pearson_coef[j,i],_=pearsonr(train_unit_number_list[j][:,i],train_RUL_list[j])
# In[]
sensor_min=[]
sensor_max=[]
for i in range(sensor_num):
    sensor_i_min=pearson_coef[:,i].min()
    sensor_i_max=pearson_coef[:,i].max()
    sensor_min.append(sensor_i_min)
    sensor_max.append(sensor_i_max)
# In[]
selected_sensor=[]
sensor_decrease=[]
sensor_increase=[]
for i in range(len(sensor_min)):
    if sensor_min[i]*sensor_max[i]>0:
        if sensor_min[i]>0.01 and sensor_max[i]>0.01:
            sensor_decrease.append(i)
            selected_sensor.append(i)
        if sensor_min[i]<-0.01 and sensor_max[i]<-0.01:
            sensor_increase.append(i)
            selected_sensor.append(i)
# In[]
file=open('selected_sensors.txt','w')
file.write("selected_sensor:"+str(selected_sensor)+"\n");
file.write("sensor_decrease:"+str(sensor_decrease)+"\n");
file.write("sensor_increase:"+str(sensor_increase)+"\n");
file.close()