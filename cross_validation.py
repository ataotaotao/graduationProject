# -*- coding: utf-8 -*-
'''
Created on 2017��9��17��

@author: Administrator
'''
import numpy as np
from sklearn import cross_validation
# from NIPALS import _NIPALS
from NIPALS import _NIPALS
from scipy.stats import f

class Cross_Validation():  # 变量初始化
    def __init__(self, x, y, n_fold, max_components):
        self.x = x
        self.y = y 
        self.n = x.shape[0]
        self.n_fold = n_fold
        self.max_components = max_components
    def cv(self):  # 划分训练集与测试集
        kf = cross_validation.KFold(self.n, self.n_fold)
     
        x_train = []
        y_train = []
        x_test = [] 
        y_test = []
        for train_index, test_index in kf:
            xtr, ytr = self.x[train_index], self.y[train_index]
            xte, yte = self.x[test_index], self.y[test_index]
            x_train.append(xtr)
            y_train.append(ytr)
            x_test.append(xte)
            y_test.append(yte)
            
        return x_train, x_test, y_train, y_test
    
    def predict_cv(self):
        x_train, x_test, y_train, y_test = self.cv()
#         print x_train
        y_allPredict = np.ones((1, self.max_components))
        pls = _NIPALS(self.max_components)
        for i in range(self.n_fold):
            y_predict = np.zeros((y_test[i].shape[0], self.max_components))
#             print 'y',np.shape(y_predict)
            x_trainMean = np.mean(x_train[i], axis=0)
            y_trainMean = np.mean(y_train[i], axis=0)
            x_testCenter = np.subtract(x_test[i], x_trainMean)
#           y_testCenter = np.subtract(y_test,y_trainMean)
            w, t, p, list_coef_B = pls.fit(x_train[i], y_train[i], self.max_components)
            for j in range(self.max_components):
#                 print "textCenter",np.shape(x_testCenter),np.shape(list_coef_B[j])
                y_pre = np.dot(x_testCenter, list_coef_B[j])
#                 print 'y_pre',np.shape(y_pre)
                y_pre = y_pre + y_trainMean
#                 print 'ravel()',np.shape(y_pre),np.shape(y_pre.ravel())
                y_predict[:, j] = y_pre.ravel()
                
            y_allPredict = np.vstack((y_allPredict, y_predict))
        y_allPredict = y_allPredict[1:]
                                                           
        return y_allPredict, self.y
    
    def mse_cv(self, y_allPredict, y_measure):
        
        
        PRESS = np.square(np.subtract(y_allPredict, y_measure))
        all_PRESS = np.sum(PRESS, axis=0)
        RMSECV = np.sqrt(all_PRESS / self.n)
        min_RMSECV = min(RMSECV)
#         print RMSECV
        comp_array = RMSECV.argsort()
#         print 'min:',comp_array
        comp_best = comp_array[0] + 1  
        return RMSECV, min_RMSECV, comp_best
    
    
    def pls2_predict_cv(self):
        x_train, x_test, y_train, y_test = self.cv()
        y_allPredict = np.ones((1, self.max_components))
        pls = _NIPALS(self.max_components)
        for i in range(self.n_fold):
#             print 'i:', i
            y_predict = np.zeros((y_test[i].shape[0], self.max_components))
#             print 'y',np.shape(y_predict)
            x_trainMean = np.mean(x_train[i], axis=0)
            y_trainMean = np.mean(y_train[i], axis=0)
            x_testCenter = np.subtract(x_test[i], x_trainMean)
#           y_testCenter = np.subtract(y_test,y_trainMean)
            w, t, p, list_coef_B = pls.fit(x_train[i], y_train[i], self.max_components)
            for j in range(self.max_components):
#                 print 'j:', j
#                 print "textCenter", np.shape(x_testCenter), np.shape(list_coef_B[j])
                y_pre = np.dot(x_testCenter, list_coef_B[j])
                y_pre = y_pre + y_trainMean
#                 print 'y_pre:', np.shape(y_pre)
                y_pre_sum = np.sum(y_pre, axis=1)
#                 print 'y_pre_sum:' np.shape(y_pre_sum)
                y_predict[:, j] = y_pre_sum.ravel()               
            y_allPredict = np.vstack((y_allPredict, y_predict))           
        y_allPredict = y_allPredict[1:]
                                                           
        return y_allPredict, self.y
    
    def pls2_mse_cv(self, y_allPredict, y_measure):
#         print 'y_allPredict:', np.shape(y_allPredict)
        PRESS = np.square(np.subtract(y_allPredict, y_measure))
        all_PRESS = np.sum(PRESS, axis=0)
        RMSECV = np.sqrt(all_PRESS / self.n)
        min_RMSECV = min(RMSECV)
        comp_array = RMSECV.argsort()
        comp_best = comp_array[0] + 1  
        return RMSECV, min_RMSECV, comp_best
        
    def cv_mse_F(self, Y_predict_all, y, alpha, num_tr):      
        press = np.square(np.subtract(Y_predict_all, y))
        PRESS_all = np.sum(press, axis=0)        
        RMSECV_array = np.sqrt(PRESS_all / self.n)
#        print RMSECV_array
        min_RMSECV = min(RMSECV_array)
        comp_array = RMSECV_array.argsort()
#        print comp_array
        comp_best = comp_array[0] + 1
#        print comp_best
        k_press = PRESS_all[:comp_best]
#        print k_press
        min_press = PRESS_all[comp_best - 1]
#        print min_press
        F_h = k_press / min_press
#        print F_h
        F_value = f.isf(alpha, num_tr, num_tr)
        F_bias = np.subtract(F_h, F_value)
#        print F_bias
        min_comp = [k for k in range(len(F_bias)) if F_bias[k] < 0]     
#         if (min_comp==0):
#             min_comp=1    
        min_comp_best = min_comp[0]
        if (min_comp_best == 0):
            min_comp_best = 1
        min_RMSECV = RMSECV_array[min_comp_best - 1]
        return  RMSECV_array, min_RMSECV, min_comp_best 
        
   #        print min_comp     
#         print  alpha
#         print num_tr
#         print F_value
    
        
        
            
