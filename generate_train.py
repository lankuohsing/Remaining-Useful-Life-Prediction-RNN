# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:11:50 2018
生成RNN训练所需数据
@author: lankuohsing
"""

# In[]
import pandas as pd
import numpy as np



# In[]
def generate_train_from_one_unit(multi_seq,TIMESTEPS=10):
    X = []
    Y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入;
    # 第i+TIMESTEPS项和后面的PREDICT_STEPS-1项作为输出
    # 即用数据的前TIMESTPES个点的信息，预测后面的PREDICT_STEPS个点的值
    for i in range(len(multi_seq) - TIMESTEPS):
        X.append(multi_seq[i:i + TIMESTEPS,0:multi_seq.shape[1]-1])
        Y.append([multi_seq[i:i + TIMESTEPS,multi_seq.shape[1]-1]])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
def generate_train_from_unit_list(unit_number_RUL_scaled_list,knee_point_np):
    # In[]
    train_X_list=[]
    train_Y_list=[]
    for i in range(len(unit_number_RUL_scaled_list)):
        # In
        unit_number_i=unit_number_RUL_scaled_list[i]#取出第i台发动机的数据
        unit_number_i_var=unit_number_i.var(axis=0)#计算各传感器的方差
        # In
        good_index_i=unit_number_i_var>-1
        unit_number_i_good=unit_number_i[:,good_index_i]
        # In
        knee_point_i=knee_point_np[i,0]
        # In
        unit_number_i_good=unit_number_i_good[knee_point_i:unit_number_i_good.shape[0],:]
        # In
        train_X_i=[]
        train_Y_i=[]
        train_X_i,train_Y_i=generate_train_from_one_unit(unit_number_i_good,TIMESTEPS=10)
        train_Y_i=np.transpose(train_Y_i,[0,2,1])
        train_X_list.append(train_X_i)
        train_Y_list.append(train_Y_i)
    return train_X_list,train_Y_list
# In[]
if __name__=="__main__":
    from read_save_data import read_unit_data
    path='unit_number_RUL_scaled.csv'
    unit_number_RUL_scaled_list=read_unit_data(path)
    # In[]
    path2='knee_point_list.csv'
    knee_point_DataFrame=pd.read_csv(path2,header=0,encoding='utf-8')
    knee_point_np=knee_point_DataFrame.as_matrix()
    # In[]
    unit_number_0=unit_number_RUL_scaled_list[0]#取出第0台发动机的数据
    unit_number_0_var=unit_number_0.var(axis=0)#计算各传感器的方差

    # In[]
    good_index_0=unit_number_0_var>-1
    # In[]
    unit_number_0_good=unit_number_0[:,good_index_0]
    # In[]
    unit_number_1=unit_number_RUL_scaled_list[1]#取出第0台发动机的数据
    unit_number_1_var=unit_number_1.var(axis=0)#计算各传感器的方差

    # In[]
    good_index_1=unit_number_1_var>-1
    # In[]
    unit_number_1_good=unit_number_1[:,good_index_1]
    # In[]
    knee_point_0=knee_point_np[0,0]
    # In[]
    unit_number_0_good=unit_number_0_good[knee_point_0:unit_number_0_good.shape[0],:]
    # In[]
    train_X=[]
    train_Y=[]
    train_X,train_Y=generate_train_from_one_unit(unit_number_0_good,TIMESTEPS=10)
    train_Y=np.transpose(train_Y,[0,2,1])

    # In[]
    train_X_list,train_Y_list=generate_train_from_unit_list(unit_number_RUL_scaled_list,knee_point_np)