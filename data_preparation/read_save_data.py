# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 19:52:34 2018

@author: lankuohsing
"""

# In[]
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# In[]
def read_train_data_and_scale(path,sensor_index,isCut=True,feature_range=(0,1)):
    """
    根据给定的路径，读取其中多台发动机的运行数据(训练集)，存在一个list中，list的每个元素是
    一台发动机的运行数据（二维矩阵）；同时将传感器数据归一化后的数据保存在csv文件中
    """
    train_DataFrame=pd.read_excel(path,sheet_name=0,header=None)#读取Excel表格中的数据
    train_np=train_DataFrame.as_matrix()  #将train数据集放在一个NumPy array中
    scaler = MinMaxScaler(copy=True,feature_range=feature_range)#copy=True保留原始数据矩阵
    train_np_scaled=scaler.fit_transform(train_np[:,5:26])
    train_np_scaled=train_np_scaled[:,sensor_index]
    train_np_scaled=np.hstack((train_np[:,0:5],train_np_scaled))
    #将归一化的数据保存在csv文件中
    np.savetxt(path.split('.')[0]+'_scaled_to_'+str(feature_range)+'_selected.csv', train_np_scaled, delimiter = ',')
    unit_number_redundant=train_np_scaled[:,0]  #提取出冗余的unit编号
    unit_number=np.unique(unit_number_redundant)  #删除unit编号中的冗余部分
    unit_nums=unit_number.shape[0]  #发动机编号数
    #将每台发动机的运行数据存在一个二维列表中，将所有的二维列表存在一个list中
    unit_number_list=[]
    for i in range(0,unit_nums):
        condition_i=train_np_scaled[:,0]==i+1#找出对应编号的数据下标集合
        unit_index_i=np.where(condition_i)
        unit_number_i_index=unit_index_i[0]
        unit_number_i=train_np_scaled[unit_number_i_index,:]
        if(isCut):
            unit_number_i=unit_number_i[:,5:unit_number_i.shape[1]]
        unit_number_list.append(unit_number_i)
    return unit_number_list,train_np,scaler.data_min_,scaler.data_max_

def read_test_data_and_scale(path,sensor_index, isCut,train_data_min_,train_data_max_):
    """
    根据给定的路径，读取其中多台发动机的运行数据（测试集），存在一个list中，list的每个元素是
    一台发动机的运行数据（二维矩阵）；同时将传感器数据归一化后的数据保存在csv文件中
    """
    #train_data_min_=np.reshape(train_data_min_,(1,len(train_data_min_)))
    #train_data_max_=np.reshape(train_data_max_,(1,len(train_data_max_)))
    test_DataFrame=pd.read_excel(path,sheet_name=0,header=None)#读取Excel表格中的数据
    test_np=test_DataFrame.as_matrix()  #将test数据集放在一个NumPy array中
    test_np_scaled=test_np
    for i in range(0,test_np.shape[1]-5):
        if train_data_min_[i]==train_data_max_[i]:
            test_np_scaled[:,5+i]=0
        else:
            test_np_scaled[:,5+i]=(test_np[:,5+i]-train_data_min_[i])/(train_data_max_[i]-train_data_min_[i])
    #test_np_scaled=(test_np[:,5:26]-train_data_min_)/(train_data_max_-train_data_min_)
    #test_np_scaled=np.hstack((test_np[:,0:5],test_np_scaled))
    #将归一化的数据保存在csv文件中
    test_np_scaled=test_np_scaled[:,list(range(5))+[i+5 for i in sensor_index]]

    np.savetxt(path.split('.')[0]+'_scaled_selected'+'.csv', test_np_scaled, delimiter = ',')
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
        if(isCut):
            unit_number_i=unit_number_i[:,5:unit_number_i.shape[1]]
        unit_number_list.append(unit_number_i)
    return unit_number_list,test_np

def read_scaled_train_data(path,isCut=True):
    """
    根据给定的路径，读取其中多台发动机的运行数据（归一化后的），存在一个list中，
    list的每个元素是一台发动机的运行数据（二维矩阵）
    """
    train_scaled_DataFrame=pd.read_csv(path,header=None,encoding='utf-8')
    train_scaled_np=train_scaled_DataFrame.as_matrix()  #将train数据集放在一个NumPy array中
    unit_number_redundant=train_scaled_np[:,0]  #提取出冗余的unit编号
    unit_number=np.unique(unit_number_redundant)  #删除unit编号中的冗余部分
    unit_nums=unit_number.shape[0]  #发动机编号数
    #将每台发动机的运行数据存在一个二维列表中，将所有的二维列表存在一个list中
    unit_number_list=[]
    for i in range(0,unit_nums):
        condition_i=train_scaled_np[:,0]==i+1#找出对应编号的数据下标集合
        unit_index_i=np.where(condition_i)
        unit_number_i_index=unit_index_i[0]
        unit_number_i=train_scaled_np[unit_number_i_index,:]
        if(isCut):
            unit_number_i=unit_number_i[:,5:unit_number_i.shape[1]]
        unit_number_list.append(unit_number_i)
    return unit_number_list
def read_unit_data(path,isCut=True):
    """
    适用于训练集和测试集
    根据给定的路径，读取其中多台发动机的运行数据，存在一个list中，
    list的每个元素是一台发动机的运行数据（二维矩阵）
    isCut:是否只提取出传感器数据
    """
    if path.split('.')[-1]=="xlsx":#判断文件类型
        print("reading xlsx......")
        unit_DataFrame=pd.read_excel(path,sheet_name=0,header=None)#读取Excel表格中的数据
    else:
        print("reading csv......")
        unit_DataFrame=pd.read_csv(path,header=None,encoding='utf-8')

    unit_np=unit_DataFrame.as_matrix()  #将数据集放在一个NumPy array中
    unit_number_redundant=unit_np[:,0]  #提取出冗余的unit编号
    unit_number=np.unique(unit_number_redundant)  #删除unit编号中的冗余部分
    unit_nums=unit_number.shape[0]  #发动机编号数
    #将每台发动机的运行数据存在一个二维列表中，将所有的二维列表存在一个list中
    unit_number_list=[]
    for i in range(0,unit_nums):
        condition_i=unit_np[:,0]==i+1#找出对应编号的数据下标集合
        unit_index_i=np.where(condition_i)
        unit_number_i_index=unit_index_i[0]
        unit_number_i=unit_np[unit_number_i_index,:]
        if(isCut):
            unit_number_i=unit_number_i[:,5:unit_number_i.shape[1]]
        unit_number_list.append(unit_number_i)
    return unit_number_list
# In[]
if __name__=="__main__":
    """
    feature_range=(0,1)#归一化的范围,类型为tuple
    path="train_FD001.xlsx"
    unit_number_list,train_np,feature_min,feature_max=read_train_data_and_scale(path)
    """
    path='train_FD001.xlsx_scaled_to_(0, 1).csv'
    unit_number_list=read_scaled_data(path,isCut=True)
