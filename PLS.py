# -*- coding: utf-8 -*-
'''
Created on 2018��1��17��

@author: Administrator
'''
from NIPALS import _NIPALS
from cross_validation import Cross_Validation
import numpy as np

class Partial_LS():
    def __init__(self, X_train, y_train, folds, max_comp):
        self.X_train = X_train
        self.y_train = y_train
        self.folds = folds
        self.max_comp = max_comp
        
    def pls_fit(self):
        
        pls_cv_demo = Cross_Validation(self.X_train, self.y_train, self.folds, self.max_comp)
#         print np.shape(self.X_train),np.shape(self.y_train)
        y_allPredict, y_measure = pls_cv_demo.predict_cv()
        
#         print 'shape:', np.shape(y_allPredict), np.shape(y_measure)
#         print y_measure
        RMSECV, min_RMSECV, comp_best = pls_cv_demo.mse_cv(y_allPredict, y_measure)
#         print RMSECV , "rmsecv"
#        RMSECV, min_RMSECV, comp_best = pls_cv_demo.cv_mse_F(y_allPredict, y_measure, alpha=0.05, num_tr=self.X_train.shape[0])
#         print "comp_best" , comp_best
        pls_demo = _NIPALS(comp_best)
        weight_W, score_T, loading_P, List_coef_B = pls_demo.fit(self.X_train, self.y_train, comp_best)
        coefs_B = List_coef_B[comp_best - 1]
#         print np.shape(coefs_B), np.shape(List_coef_B), comp_best
        
        return weight_W, score_T, loading_P, comp_best, coefs_B, RMSECV
    def pls_pre(self, X_test, y_test, coefs_B):
        
        Xtrain_center = np.subtract(self.X_train, self.X_train.mean(axis=0))
#         drawing.spectrum(Xtrain_center)
        ytrain_pre_center = np.dot(Xtrain_center, coefs_B)
        ytrain_pre = np.add(ytrain_pre_center, self.y_train.mean(axis=0))
        
        press = np.square(np.subtract(self.y_train, ytrain_pre))
        all_press = np.sum(press, axis=0)
        RMSEC = np.sqrt(all_press / self.X_train.shape[0])
        
        
        
        Xtest_center = np.subtract(X_test, self.X_train.mean(axis=0))
        ytest_pre_center = np.dot(Xtest_center, coefs_B)
        ytest_pre = np.add(ytest_pre_center, self.y_train.mean(axis=0))
        
        press = np.square(np.subtract(y_test, ytest_pre))
        all_press = np.sum(press, axis=0)
        RMSEP = np.sqrt(all_press / X_test.shape[0])
        
        return RMSEC, RMSEP ,ytest_pre 
    
    def pls_pre_train_test(self, X_test, y_test, coefs_B):
        
        Xtrain_center = np.subtract(self.X_train, self.X_train.mean(axis=0))
#         drawing.spectrum(Xtrain_center)
        ytrain_pre_center = np.dot(Xtrain_center, coefs_B)
        ytrain_pre = np.add(ytrain_pre_center, self.y_train.mean(axis=0))
        
        press = np.square(np.subtract(self.y_train, ytrain_pre))
        all_press = np.sum(press, axis=0)
        RMSEC = np.sqrt(all_press / self.X_train.shape[0])
        
        
        
        Xtest_center = np.subtract(X_test, self.X_train.mean(axis=0))
        ytest_pre_center = np.dot(Xtest_center, coefs_B)
        ytest_pre = np.add(ytest_pre_center, self.y_train.mean(axis=0))
        
        press = np.square(np.subtract(y_test, ytest_pre))
        all_press = np.sum(press, axis=0)
        RMSEP = np.sqrt(all_press / X_test.shape[0])
        
        return RMSEC, RMSEP ,ytrain_pre,ytest_pre 
    def pls_pre_train(self, X_test, y_test, coefs_B):
        
        Xtrain_center = np.subtract(self.X_train, self.X_train.mean(axis=0))
#         drawing.spectrum(Xtrain_center)
        ytrain_pre_center = np.dot(Xtrain_center, coefs_B)
        ytrain_pre = np.add(ytrain_pre_center, self.y_train.mean(axis=0))
        
        press = np.square(np.subtract(self.y_train, ytrain_pre))
        all_press = np.sum(press, axis=0)
        RMSEC = np.sqrt(all_press / self.X_train.shape[0])
        
        Xtest_center = np.subtract(X_test, self.X_train.mean(axis=0))
        ytest_pre_center = np.dot(Xtest_center, coefs_B)
        ytest_pre = np.add(ytest_pre_center, self.y_train.mean(axis=0))
        
        press = np.square(np.subtract(y_test, ytest_pre))
        all_press = np.sum(press, axis=0)
        RMSEP = np.sqrt(all_press / X_test.shape[0])
        
        return RMSEC, RMSEP , ytrain_pre 
    def pls2_fit(self):
        
        pls2_cv_demo = Cross_Validation(self.X_train, self.y_train, self.folds, self.max_comp)
        
        y_allPredict, y_measure = pls2_cv_demo.pls2_predict_cv()
        y = np.sum(y_measure, axis=1)
        y = y.reshape(self.X_train.shape[0], 1)
#         print 'measure:', np.shape(y), y
        RMSECV, min_RMSECV, comp_best = pls2_cv_demo.pls2_mse_cv(y_allPredict, y)
        
#        RMSECV, min_RMSECV, comp_best = pls_cv_demo.cv_mse_F(y_allPredict, y_measure, alpha=0.05, num_tr=self.X_train.shape[0])
        
        pls2_demo = _NIPALS(comp_best)
        weight_W, score_T, loading_P, List_coef_B = pls2_demo.fit(self.X_train, self.y_train, comp_best)
        coefs_B = List_coef_B[comp_best - 1]
#         print 'infor:', comp_best, np.shape(self.X_train), np.shape(self.y_train), np.shape(List_coef_B), np.shape(coefs_B)
#         print comp_best , "comp_best"
#         print coefs_B.shape , "shape"
        return weight_W, score_T, loading_P, comp_best, coefs_B, RMSECV
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    from scipy.io.matlab.mio import loadmat
    from sklearn.cross_validation import train_test_split
    
    fname = loadmat('NIRcorn.mat')
    print fname.keys()
    X_master = fname['cornspect']
    y = fname['cornprop'][:, 1:2]
#     X_slave = fname['mp6spec']['data'][0][0]
    
#     print np.shape(X_master)    print np.shape(X_slave)    print np.shape(y)
#     (80, 700)    (80, 700)    (80, 1)
    folds = 10
    max_comp = 30
    
    X_m_cal, X_m_test, y_m_cal, y_m_test = train_test_split(X_master, y, test_size=0.2, random_state=0)
#     X_s_cal, X_s_test, y_s_cal, y_s_test = train_test_split(X_slave, y, test_size=0.2, random_state=0)
    
#     X_m_train, X_m_std, y_m_std, y_m_std = train_test_split(X_m_cal, y_m_cal, test_size=0.5, random_state=0)
#     X_s_train, X_s_std, y_s_std, y_s_std = train_test_split(X_s_cal, y_s_cal, test_size=0.5, random_state=0)
#    
    print np.shape(X_m_cal), np.shape(y_m_cal)
    demo = Partial_LS(X_m_cal, y_m_cal, folds, max_comp)
    W_slave_cal, T_slave_cal, P_slave_cal, comp_best_slave_cal, coefs_slave_cal, RMSECV_slave_cal = demo.pls_fit()
    pls_RMSEC, pls_RMSEP = demo.pls_pre(X_m_test, y_m_test, coefs_slave_cal)
    print 'comp_best', comp_best_slave_cal
    print 'RMSEC', pls_RMSEC
    print 'RMSEP', pls_RMSEP








