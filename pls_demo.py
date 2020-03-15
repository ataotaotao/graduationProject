# *-coding:utf-8-*-
'''
Created on 2015骞�1鏈�鏃�

@author: lenovo
'''
from __future__ import division
from sklearn import cross_validation 
# from sklearn.decomposition import PCA
import pylab as pl
import scipy.io as sio
import numpy as np

from NIPALS import _NIPALS
from cross_val import PLSCV
# import drawings 
# import time


class PlsDemo():
    def __init__(self, x_train, y_train, n_folds, max_components):
        
        self.x_train = x_train
        self.y_train = y_train
        
        self.n_folds = n_folds
        self.max_components = max_components
        
    def pls_fit(self):
        
        pls_cv = PLSCV(self.x_train, self.y_train)
        y_predict_all, y_measure = pls_cv.cv_predict(self.n_folds, self.max_components)
            
        RMSECV, min_RMSECV, comp_best = pls_cv.cv_mse(y_predict_all, y_measure)
#        print RMSECV
        pls = _NIPALS(comp_best)
            
        
        W, T, P, lists_coefs = pls.fit(self.x_train, self.y_train, comp_best)
#        print self.y_train
        coefs_B = lists_coefs[comp_best - 1]
#        drawings.rmsecv_comp_line(self.max_components, RMSECV)
        
        return W, T, P, coefs_B, RMSECV, min_RMSECV, comp_best
        
    def pls_predict(self, x_test, y_test, coefs_B):
        
        xtr_mean = np.mean(self.x_train, axis=0)
        ytr_mean = np.mean(self.y_train, axis=0)
    
#        yte_predict=pls.predict(self.x_test,coefs_B,xtr_mean,ytr_mean)
        xtr_center = np.subtract(self.x_train, xtr_mean)
        ytr_pre = np.dot(xtr_center, coefs_B)
        ytr_predict = np.add(ytr_pre, ytr_mean)
        RMSEC = np.sqrt(np.sum(np.square(np.subtract(self.y_train, ytr_predict)), axis=0) / self.x_train.shape[0])
        print '%.4f' % RMSEC[0]
#         print RMSEC
        xte_center = np.subtract(x_test, xtr_mean)
        y_pre = np.dot(xte_center, coefs_B)
        yte_predict = np.add(y_pre, ytr_mean)
        RMSEP = np.sqrt(np.sum(np.square(np.subtract(y_test, yte_predict)), axis=0) / x_test.shape[0])

        bias = np.sum(np.subtract(y_test, yte_predict), axis=0) / x_test.shape[0]
        SEP = np.sum(np.square(np.subtract(y_test, yte_predict) - bias), axis=0) / (x_test.shape[0] - 1)
###################################################SEP?????????????????????????????????????????????
        return yte_predict, RMSEP, SEP
        
        
    
    
    
if __name__ == '__main__':
    import numpy as np
    from scipy import linalg
    from scipy.io import loadmat, savemat
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    
    
    fname = loadmat('Pharmaceutical tablet')
    D = fname
    print D.keys()
             
    x_src_cal = D['calibrate_1']['data'][0][0]
    x_tar_cal = D['calibrate_2']['data'][0][0]
    y_src_cal = D['calibrate_Y']['data'][0][0][:, 2:3]
    y_tar_cal = y_src_cal   
#     x_src_std=D['validate_1']['data'][0][0]
#     x_tar_std=D['validate_2']['data'][0][0]
              
    x_src_test = D['test_1']['data'][0][0]
    x_tar_test = D['test_2']['data'][0][0]
    y_src_test = D['test_Y']['data'][0][0][:, 2:3]
    y_tar_test = y_src_test

 
#     fname = loadmat('NIRcorn.mat')
#     D = fname
#     print D.keys()
#                         
#     x = D['cornspect']
#             
#     x_mp5spec=D['mp5spec']['data'][0][0]
#     x_m5spec=D['m5spec']['data'][0][0]   #x_m5spec=x
#     x_mp6spec=D['mp6spec']['data'][0][0]
#     x_src=x_m5spec
#     x_tar=x_mp6spec
#         
#     y = D['cornprop'][:,3:4] 
#     x_src_cal,x_src_test,y_src_cal,y_src_test=train_test_split(x_src,y,test_size=0.2,random_state=100)     
#     x_tar_cal,x_tar_test,y_tar_cal,y_tar_test=train_test_split(x_tar,y,test_size=0.2,random_state=100)
     
    n_folds = 10
    max_components = 5

    demo = PlsDemo(x_src_cal, y_src_cal, n_folds, max_components)
#         
    W, T, P, coefs_B, RMSECV, min_RMSECV, comp_best = demo.pls_fit()
    yte_predict, RMSEP, SEP = demo.pls_predict(x_src_test, y_src_test, coefs_B)
    
    
    print '%.4f' % min_RMSECV
    print '%.4f' % RMSEP
    print comp_best
#     demo1=PlsDemo(x_tar_cal,y_tar_cal,n_folds,max_components)
# #         
#     W1,T1,P1,coefs_B1,RMSECV1,min_RMSECV1,comp_best1=demo1.pls_fit()
#     yte_predict1,RMSEP1,SEP1=demo1.pls_predict(x_tar_test,y_tar_test,coefs_B1)
#     print RMSEP1

    
    

    
 
