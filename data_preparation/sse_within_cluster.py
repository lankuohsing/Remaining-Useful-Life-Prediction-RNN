# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:07:00 2017

@author: lankuohsing
"""

# In[]
import numpy as np
# In[]
def get_centroid(points):
    """
    calculate the centroid of given points
    计算给定点集的几何中心
    """
    points_sum=np.sum(points,axis=0)
    points_mean=points_sum/points.shape[0]
    return points_mean
def get_SSE(points):
    """
    calculate the sum of squared errors within a cluster
    计算簇内误差平方和
    """
    points_mean=get_centroid(points)
    points_deviation=points-points_mean
    points_deviation_norm=np.linalg.norm(points_deviation,ord=2,axis=1,keepdims=True)
    points_sse=np.sum(points_deviation_norm)/points_deviation_norm.shape[0]
    return points_sse
if __name__=="__main__":

    points=np.array([[1,2],[3,4],[5,6]])
    print(get_SSE(points))

