# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:33:18 2018

@author: lankuohsing
"""
# In[]
import string
# In[]
f=open('RUL_FD001.txt')
a=f.read()
print(a)
f.close()
b=a.split('\n')
c=b[100]
if(c==''):
    print("bingo")
d=[]
# In[]
for i in range(len(b)):
    if (b[i]!=''):
        d.append(int(b[i]))
