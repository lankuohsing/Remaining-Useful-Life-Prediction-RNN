# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:59:40 2018

@author: lankuohsing
"""
# In[]
import numpy as np
import matplotlib.pyplot as plt
from read_save_data import read_train_data_and_scale
from sse_within_cluster import get_centroid
# In[]
def get_between_cluster_se(unit_number_scaled_0,window=5):
    #unit_number_scaled_0=unit_number_list[j]#选取第0个编号的发动机做一下试验
    between_cluster_se_0=[]
    between_cluster_se_list=[]

    for i in range(0,unit_number_scaled_0.shape[0]//window):
        unit_number_scaled_0_i_1=unit_number_scaled_0[window*0:window*(0+1),:]
        unit_number_scaled_0_i_2=unit_number_scaled_0[window*(i):window*(i+1),:]

        mean_0_1=get_centroid(unit_number_scaled_0_i_1)
        mean_0_2=get_centroid(unit_number_scaled_0_i_2)
        between_cluster_se_0_i=np.sum(np.square(mean_0_1-mean_0_2))
        between_cluster_se_0.append(between_cluster_se_0_i)
        between_cluster_se_list.append(between_cluster_se_0_i)
    between_cluster_se_dot_list=[]
    for i in range(0,len(between_cluster_se_list)-1):
        between_cluster_se_dot_i=(between_cluster_se_list[i+1]-
                                  between_cluster_se_list[i])/(window)*1000
        between_cluster_se_dot_list.append(between_cluster_se_dot_i)
    for i in range(0,len(between_cluster_se_dot_list)):
        between_cluster_se_dot_list[i]=between_cluster_se_dot_list[i]/between_cluster_se_dot_list[-1]
    return between_cluster_se_list,between_cluster_se_dot_list
def get_knee_point(between_cluster_se_list,between_cluster_se_dot_list,dot_threshold=0.05,window=10):

    knee_point=0
    for i in range(0,len(between_cluster_se_dot_list)):
        if(between_cluster_se_dot_list[i]>dot_threshold) \
        and (between_cluster_se_list[i+2]>between_cluster_se_list[i+1])\
        and (between_cluster_se_list[i+3]>between_cluster_se_list[i+2]):
            knee_point=window*i+window//2
            break
    return knee_point
# In[]
if __name__=="__main__":
    path="train_FD001.xlsx"
    feature_range=(0,1)#归一化的范围,类型为tuple
    unit_number_list,raw_data,feature_min,feature_max=read_data_and_scale(path)
    # In[]
    j=1
    unit_number_scaled_0=unit_number_list[j]#选取第0个编号的发动机做一下试验
    # In
    window=10
    dot_threshold=0.05
    between_cluster_se_list,between_cluster_se_dot_list=\
    get_between_cluster_se(unit_number_scaled_0,window=window)
    plt.figure(1)
    #plt.subplot(211)
    plot_between_cluster_se=plt.plot(between_cluster_se_list,'r.')
    #plt.subplot(212)
    plt.figure(2)
    plot_between_cluster_se_dot=plt.plot(between_cluster_se_dot_list,'b.')
    plt.show()
    # In
    knee_point=\
    get_knee_point(between_cluster_se_list,
                   between_cluster_se_dot_list,
                   dot_threshold=dot_threshold,
                   window=window)
    print(knee_point)
