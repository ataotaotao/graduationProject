# -*- coding: utf-8 -*-
'''
Created on 2018��1��27��

@author: Administrator
'''
from numpy import arange, array, zeros, where
import numpy as np
from numpy.linalg import inv, norm

class KennardStone():
    def __init__(self, X, Num):
        self.x = X
        self.num = Num
        
    def KS(self):
        nrow = self.x.shape[0]
        CalInd = zeros((self.num), dtype=int) - 1
        vAll = arange(0, nrow)
        D = zeros((nrow, nrow))
        for i in range(nrow - 1):
            for j in range(i + 1, nrow):
                D[i, j] = norm(self.x[i, :] - self.x[j, :])
        ind = where(D == D.max())
        CalInd[0] = ind[1]
        CalInd[1] = ind[0]
        for i in range(2, self.num):
            vNotSelected = array(list(set(vAll) - set(CalInd)))
            vMinDistance = zeros(nrow - i)
            for j in range(nrow - i):
                nIndexNotSelected = vNotSelected[j]
                vDistanceNew = zeros((i))
                for k in range(i):
                    nIndexSelected = CalInd[k]
                    if nIndexSelected <= nIndexNotSelected:
                        vDistanceNew[k] = D[nIndexSelected, nIndexNotSelected]
                    else:
                        vDistanceNew[k] = D[nIndexNotSelected, nIndexSelected]
                vMinDistance[j] = vDistanceNew.min()
            nIndexvMinDistance = where(vMinDistance == vMinDistance.max())
            CalInd[i] = vNotSelected[nIndexvMinDistance]
        ValInd = array(list(set(vAll) - set(CalInd)))
        return CalInd, ValInd


if __name__ == '__main__':
    from scipy.io import loadmat
    fname = loadmat('NIRcorn.mat')
    print fname.keys()
    X_master = fname['cornspect']
    y = fname['cornprop'][:, 0:1]
    X_slave = fname['mp6spec']['data'][0][0]
    KS_demo = KennardStone(X_master, 5)
    CalInd, ValInd = KS_demo.KS()
    
    X_master_cal = X_master[CalInd]
    X_master_test = X_master[ValInd]
    
#     print np.shape(X_master_cal)
#     print np.shape(X_master_test)
#     print X_master_cal
    
    
    
    
