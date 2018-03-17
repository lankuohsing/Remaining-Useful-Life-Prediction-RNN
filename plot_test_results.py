# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 20:19:33 2018

@author: lankuohsing
"""
# In[]
import numpy as np
import matplotlib.pyplot as plt
# In[]
S_list_1_256_25=[7938.778234988969, 8043.938608396231, 7845.124958448783, 7711.340344370845,
                 7575.591452613173, 1669.992856806489, 8026.765173563922, 2159.3575423966054,
                 1996.3627516456222, 7578.789943439138, 1286.0262030883089, 1204.9614613939639,
                 1060.7126777328165, 1109.2662592432216, 1107.7725069996654, 1209.2646345896649,
                 1042.7497697269864, 1128.2477009984768, 871.3352147289203, 784.4354591314998,

                 7949.114859403331, 8013.063413023013, 7866.768626174545, 7711.491691139294,
                 7580.073111643874, 1296.2373882229417, 5431.157300683879, 7849.7503520867485,
                 5280.815436978941, 3379.0753640326643, 1165.4902010305598, 986.9385497895842,
                 872.9254290654676, 1085.3838608999965, 798.8998316558319, 1032.8840217218078,
                 1560.8658596233627, 758.7862698361321, 742.8275590694046, 819.2805894404161]
S_np=np.array(S_list_1_256_25).reshape((2,4,5))
# In[]
num_layers_np=np.array([1,2])
lstm_size_np=np.array([32,64,128,256])
num_steps_np=np.array([5,10,15,20,25])
linestyle=['cx--','mo:','kp-.','bs--','p*:'] #红，绿，黄，蓝，粉,每个折线给不同的颜色
lstm_size_list=['32','64','128','256']
for i in range(S_np.shape[0]):

    plt.figure(i,figsize=(12,8))
    plt.title('Test Result Analysis:num_layers_np='+str(num_layers_np[i]),fontsize=20)
    for j in range(S_np.shape[1]):

        plt.plot(num_steps_np,S_np[i,j,:],linestyle[j],label='lstm_size='+lstm_size_list[j])
        plt.tick_params(labelsize=20)
    plt.legend(fontsize=10) # 显示图例
    plt.xlabel("num_steps",fontsize=20)
    plt.ylabel("score",fontsize=20)
    plt.savefig('plot_test/'+'num_layers='+str(num_layers_np[i])+
                    '_lstm_size='+str(lstm_size_np[j])+'.png')
    plt.show()

# In[]
