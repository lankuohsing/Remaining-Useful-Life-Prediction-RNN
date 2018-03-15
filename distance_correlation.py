# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:31:46 2018

@author: lankuohsing
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:00:29 2012
Author: Josef Perktold
License: MIT, BSD-3 (for statsmodels)
http://en.wikipedia.org/wiki/Distance_correlation
Yaroslav and Satrajit on sklearn mailing list
Univariate only, distance measure is just absolute distance
Note: Same as R package energy DCOR, except DCOR reports sqrt of all returns of dcov_all
"""

import numpy as np

from scipy.stats import pearsonr
def dist(x, y):
    #1d only
    return np.abs(x[:, None] - y)


def d_n(x):
    d = dist(x, x)
    dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
    return dn


def dcov_all(x, y):
    dnx = d_n(x)
    dny = d_n(y)

    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx**2).sum() / denom
    dvy = (dny**2).sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return dc, dr, dvx, dvy


import matplotlib.pyplot as plt

fig = plt.figure()
for case in range(1,5):

    np.random.seed(9854673)
    x = np.linspace(-1,1, 501)
    if case == 1:
        y = - x**2 + 0.2 * np.random.rand(len(x))
    elif case == 2:
        y = np.cos(x*2*np.pi) + 0.1 * np.random.rand(len(x))
    elif case == 3:
        x = np.sin(x*2*np.pi) + 0.0 * np.random.rand(len(x))  #circle
    elif case == 4:
        x = np.sin(x*1.5*np.pi) + 0.1 * np.random.rand(len(x))  #bretzel
    dc, dr, dvx, dvy = dcov_all(x, y)
    print( dc, dr, dvx, dvy)
    per_coe,_= pearsonr(x, y)
    print(per_coe)
    ax = fig.add_subplot(2,2, case)
    #ax.set_xlim(-1, 1)
    ax.plot(x, y, '.')
    yl = ax.get_ylim()
    ax.text(-0.95, yl[0] + 0.9 * np.diff(yl), 'dr=%4.2f' % dr)

plt.show()

# In[]
x = np.linspace(0,2, 501)
y=x*x-x
dc, dr, dvx, dvy = dcov_all(x, y)
print( dc, dr, dvx, dvy)
per_coe,_= pearsonr(x, y)
print(per_coe)