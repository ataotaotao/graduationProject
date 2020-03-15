# -*- coding: utf-8 -*-
'''
Created on 2018��4��14��
     仿射变换，预处理  采 用归一化,标准化方法
     论文
@author: Administrator
'''


from __future__ import division
from sklearn import cross_validation 
from sklearn.decomposition import PCA
from sklearn import linear_model
# import pylab as pl
import scipy.io as sio
from PLS import Partial_LS
import numpy as np
from scipy import linalg
import math
from sklearn.preprocessing import MinMaxScaler, scale
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from NIPALS import _NIPALS
from pls_demo import PlsDemo
from KS_algorithm import KennardStone
import function_module as fm

class SBC():
    
    def __init__(self, x_src_cal, y_src_cal, x_m_std, y_m_std, x_tar_std, y_tar_std, x_tar_test, y_tar_test, n_folds, max_components):
        
        self.x_src_cal = x_src_cal
        self.y_src_cal = y_src_cal
        self.x_m_std = x_m_std
        self.y_m_std = y_m_std
        
        self.x_tar_std = x_tar_std
        self.y_tar_std = y_tar_std
        self.x_tar_test = x_tar_test
        self.y_tar_test = y_tar_test
        self.n_folds = n_folds
        self.max_components = max_components
        
    def transform(self):
        
        PD_src = PlsDemo(self.x_src_cal, self.y_src_cal, self.n_folds, self.max_components)
        W_src, T_src, P_src, coefs_B_src, RMSECV_src, min_RMSECV_src, comp_best_src = PD_src.pls_fit()
        
        x_src_mean = np.mean(self.x_src_cal, axis=0)
        y_src_mean = np.mean(self.y_src_cal, axis=0)
        x_tar_center = np.subtract(self.x_tar_std, x_src_mean)
        y_tar_pre = np.add(np.dot(x_tar_center, coefs_B_src), y_src_mean)         
#         print ':', np.dot(self.x_m_std, coefs_B_src)
        
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit (y_tar_pre, self.y_tar_std)
        bias_y = clf.coef_
       
        return bias_y, coefs_B_src, comp_best_src  # , bia_list, sin_x_list, cos_x_list , x_list, P_m_std, T_m_std, T_s_std


    def affine_predict(self, T_s_test, bia_list, sin_x_list, cos_x_list):
        
        n, m = np.shape(T_s_test)
#         print 'n,m', n, m
        T_trans_test = np.zeros((n, m))
#         print 'T_s_std:', T_s_test
#         print self.y_tar_std
#         print cos_x_list
#         print sin_x_list
#         print bia_list
        for j in range(m):
            t_list = []
            for i in range(n):
                t = T_s_test[i:i + 1, j:j + 1]
                y = self.y_tar_test[i]
#                 print 't,y', t, y
                y = y + bia_list[j]
                t_trans = t * cos_x_list[j] - y * sin_x_list[j]
                t_list.append(t_trans)
#             print 'T_list:', t_list   
#             print 'shape:', np.shape(T_trans_test[:, j:j + 1]), np.shape(np.array(t_list).reshape(-1, 1))
            T_trans_test[:, j:j + 1] = np.array(t_list).reshape(-1, 1)
#         print 'y_fin_pre', y_fin_pre
        return T_trans_test  
        
    def predict(self, bias_y, coefs_B_src,x,y):
        
        x_src_mean = np.mean(self.x_src_cal, axis=0)
        y_src_mean = np.mean(self.y_src_cal, axis=0) 
        x_tar_center = np.subtract(x, x_src_mean)
        y_tar_pre = np.add(np.dot(x_tar_center, coefs_B_src), y_src_mean)
        
        y_tar_predict = np.dot(y_tar_pre, bias_y)
                
        RMSEP = np.sqrt(np.sum(np.square(np.subtract(y_tar_predict, y)), axis=0) / x.shape[0])

        return RMSEP, y_tar_predict
    
    

class Affine_trans():
    def __init__(self, T_m_std, T_s_std, y_m_pre, y_s_pre, comp_best):
        
        self.T_m_std = T_m_std
        self.T_s_std = T_s_std
        self.y_m_pre = y_m_pre
        self.y_s_pre = y_s_pre
        self.comp_best = comp_best
        
    def affine_trans(self, Tm_std_norm, Ts_std_norm, y_m_pre_norm, y_s_pre_norm, draw=True):

        Max_m = np.max(Tm_std_norm, axis=0)
        Max_s = np.max(Ts_std_norm, axis=0)
        Min_m = np.min(Tm_std_norm, axis=0)
        Min_s = np.min(Ts_std_norm, axis=0)
#         print 'max min: ', Max_m, Max_s, Min_m, Min_s
        if Max_m > Max_s:
            max_value = Max_m
        else:
            max_value = Max_s
           
        if Min_m > Min_s:
            min_value = Min_s
        else:
            min_value = Min_m
        clf = linear_model.LinearRegression(fit_intercept=True)
        clf.fit (Tm_std_norm.reshape(-1, 1), y_m_pre_norm.reshape(-1, 1))
        k_m_std = clf.coef_
        b_m_std = clf.intercept_
        
        clf = linear_model.LinearRegression(fit_intercept=True)
        clf.fit (Ts_std_norm.reshape(-1, 1), y_s_pre_norm.reshape(-1, 1))
        k_s_std = clf.coef_
        b_s_std = clf.intercept_   
             
#         print 'k,b', k_m_std, b_m_std
#         print k_s_std, b_s_std   
          
        T = np.array([min_value, max_value]).reshape(-1, 1)
        y_m_start = k_m_std * T[0] + b_m_std
        y_m_end = k_m_std * T[1] + b_m_std
        y_s_start = k_s_std * T[0] + b_s_std
        y_s_end = k_s_std * T[1] + b_s_std
        y_m = np.array([y_m_start, y_m_end]).reshape(-1, 1)
        y_s = np.array([y_s_start, y_s_end]).reshape(-1, 1)
        
        if draw == True:
            drawing.line_fit(T, y_m, y_s, Tm_std_norm, Ts_std_norm, y_m_pre_norm, y_s_pre_norm)
            
        bia = b_m_std - b_s_std
    
            # x1,y1为真实直线的向量形式            x2,y2为预测直线的向量形式
            
        x1 = T[1] - T[0]
        y1 = y_m[1] - y_m[0]
        x2 = T[1] - T[0]
        y2 = y_s[1] - y_s[0]
    #         print x1, y1, x2, y2
    
        start_length = np.sqrt(np.square(x1) + np.square(y1))
        end_length = np.sqrt(np.square(x2) + np.square(y2))
        cos_x = np.array((x1 * x2) + (y1 * y2)) / np.array(start_length * end_length)
    
    #         print math.acos(cos_x)
        x = math.degrees(math.acos(cos_x)) 
        sin_x = math.sin(math.radians(x))
    
        if y1 < y2:
            x = -1 * x
            sin_x = math.sin(math.radians(x))
            cos_x = math.cos(math.radians(x))
            
    #         print 'infor', x, sin_x, cos_x, bia        
        return bia, sin_x, cos_x, x, b_s_std   

    def AT_train(self):
        bia_list = []
        sin_x_list = []
        cos_x_list = []
        x_list = []
        b_s_list = []
        y_m_pre_norm = fm.Norm(self.y_m_pre)
        y_s_pre_norm = fm.Norm(self.y_s_pre)
        for i in range(self.comp_best):
            Tm_std_norm = fm.Norm(self.T_m_std[:, i:i + 1])
            Ts_std_norm = fm.Norm(self.T_s_std[:, i:i + 1])
            bia, sin_x, cos_x, x, b_s = self.affine_trans(Tm_std_norm, Ts_std_norm, y_m_pre_norm, y_s_pre_norm, draw=False)           
            
            bia_list.append(bia)
            sin_x_list.append(sin_x)
            cos_x_list.append(cos_x)
            x_list.append(x)      
            b_s_list.append(b_s)
        return bia_list, sin_x_list, cos_x_list, x_list, b_s_list
    
    def AT_T0y_fit(self):
        bia_list = []
        sin_x_list = []
        cos_x_list = []
        x_list = []
        b_s_list = []
        y_m_pre_norm = fm.Norm(self.y_m_pre)
        y_s_pre_norm = fm.Norm(self.y_s_pre)
        for i in range(self.comp_best):
            Tm_std_norm = fm.Norm(self.T_m_std[:, i:i + 1])
            Ts_std_norm = fm.Norm(self.T_s_std[:, i:i + 1])
            bia, sin_x, cos_x, x, b_s = self.affine_trans(Tm_std_norm, Ts_std_norm, self.T_m_std[:, 0:1], self.T_s_std[:, 0:1], draw=False)           
            
            bia_list.append(bia)
            sin_x_list.append(sin_x)
            cos_x_list.append(cos_x)
            x_list.append(x)      
            b_s_list.append(b_s)
        return bia_list, sin_x_list, cos_x_list, x_list, b_s_list
    
    
    def AT_fit(self, T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, b_s_list, y_s_test):
        
        T_trans_test, y_trans_test = self.transform(T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, b_s_list)
#         print 'y_trans_test:', y_trans_test
#         print 'y_s_test:', y_s_test
        
        y_s_predict = np.array(np.mean(y_trans_test, axis=1)).reshape(-1, 1)
        
        PRESS = np.square(np.subtract(y_s_predict, y_s_test))
        all_press = np.sum(PRESS, axis=0)
        RMSEP = np.sqrt(all_press / y_s_test.shape[0]) 
#         print 'train-error:', RMSEP
        
        clf = linear_model.LinearRegression(fit_intercept=True)
        clf.fit (y_s_predict.reshape(-1, 1), y_s_test.reshape(-1, 1))
        k = clf.coef_
        b = clf.intercept_
 
        return k, b
    
    def CV_train(self, Tm_train, Ts_train, ym_train, ys_train):
        bia_list = []
        sin_x_list = []
        cos_x_list = []
        x_list = []
        b_s_list = []
        y_m_pre_norm = fm.Norm(ym_train)
        y_s_pre_norm = fm.Norm(ys_train)
        for i in range(self.comp_best):
            Tm_std_norm = fm.Norm(Tm_train[:, i:i + 1])
            Ts_std_norm = fm.Norm(Ts_train[:, i:i + 1])
            bia, sin_x, cos_x, x, b_s = self.affine_trans(Tm_std_norm, Ts_std_norm, y_m_pre_norm, y_s_pre_norm, draw=False)           
            
            bia_list.append(bia)
            sin_x_list.append(sin_x)
            cos_x_list.append(cos_x)
            x_list.append(x)      
            b_s_list.append(b_s)
        return bia_list, sin_x_list, cos_x_list, x_list, b_s_list
    
    def CV_pre(self, T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, b_s_list, Tm_train, ym_train, Ts_train, ys_train):

        n, m = np.shape(T_s_test)
        T_trans_test_norm = np.zeros((n, m))
        y_trans_test_norm = np.zeros((n, m))
        
        scaler_T = MinMaxScaler((0, 1)).fit(Ts_train)
        Ts_test_norm = scaler_T.transform(T_s_test)
#         print 'Ts_test_norm:', Ts_test_norm       
        scaler_y = MinMaxScaler((0, 1)).fit(ys_train)
        ys_test_pre_norm = scaler_y.transform(ys_test_pre)
#         print 'ys_test_pre_norm:', ys_test_pre_norm      
        for j in range(m):
            t_list = []
            y_list = [] 
            for i in range(n):
                t = Ts_test_norm[i, j]
                y = ys_test_pre_norm[i]
             
                t_trans = t * cos_x_list[j] - (y - b_s_list[j]) * sin_x_list[j]  #################################
                y_trans = t * sin_x_list[j] + (y - b_s_list[j]) * cos_x_list[j] + b_s_list[j]  #################################
                y_trans = y_trans + bia_list[j]
                
                t_list.append(t_trans)
                y_list.append(y_trans)
                
            T_trans_test_norm[:, j:j + 1] = np.array(t_list).reshape(-1, 1)
            y_trans_test_norm[:, j:j + 1] = np.array(y_list).reshape(-1, 1)
        
        scaler_T = MinMaxScaler((0, 1)).fit(Tm_train)
        T_trans_test = scaler_T.inverse_transform(T_trans_test_norm)
        
        scaler_y = MinMaxScaler((0, 1)).fit(ym_train)
        y_trans_test = scaler_y.inverse_transform(y_trans_test_norm)
        
        return T_trans_test, y_trans_test
  
    def cross_val(self):
        ########################交叉验证，找到误差最小的一列
        Tm_train, Tm_test, ym_train, ym_test = fm.cv_split_data(self.T_m_std, self.y_m_pre, 10)
        Ts_train, Ts_test, ys_train, ys_test = fm.cv_split_data(self.T_s_std, self.y_s_pre, 10)
        
        y_allPre = np.ones((1, self.comp_best))
        for i in range(10):
            bia_list, sin_x_list, cos_x_list, x_list, b_s_list = self.CV_train(Tm_train[i], Ts_train[i], ym_train[i], ys_train[i])
            T_trans_test, y_trans_test = self.CV_pre(Ts_test[i], ys_test[i], bia_list, cos_x_list, sin_x_list, b_s_list, Tm_train[i], ym_train[i], Ts_train[i], ys_train[i])
            y_allPre = np.vstack((y_allPre, y_trans_test))
        
        y_allPre = y_allPre[1:]    
           
        return y_allPre, self.y_m_pre
    
    def cv_choose_para(self, y_allPre, y_measure):
        
        n = self.T_m_std.shape[0]
        PRESS = np.square(np.subtract(y_allPre, y_measure))
        all_PRESS = np.sum(PRESS, axis=0)
        RMSECV = np.sqrt(all_PRESS / n)
        min_RMSECV = min(RMSECV)
        comp_array = RMSECV.argsort()
        comp_best = comp_array[0] 
        
        return RMSECV, min_RMSECV, comp_best
    
    def AT_pre(self, T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, y_s_test , b_s_list):
        
        y_allPre, y_measure = self.cross_val()
        RMSECV, min_RMSECV, comp_best = self.cv_choose_para(y_allPre, y_measure)
#         print RMSECV
#         drawing.line_rmsecv_comp(self.comp_best, RMSECV)
#         print 'RMSECV:', RMSECV
#         print 'feature_comp_best:', comp_best
        
        T_trans_test, y_trans_test = self.transform(T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, b_s_list)
        y_s_predict = np.array(np.mean(y_trans_test, axis=1)).reshape(-1, 1)

        PRESS = np.square(np.subtract(y_s_predict, y_s_test))
        all_press = np.sum(PRESS, axis=0)
        RMSEP = np.sqrt(all_press / y_s_test.shape[0]) 
        
        PRESS = np.square(np.subtract(y_trans_test[:, 0:1], y_s_test))
        all_press = np.sum(PRESS, axis=0)
        RMSEP_t0 = np.sqrt(all_press / y_s_test.shape[0]) 
        
        PRESS = np.square(np.subtract(y_trans_test[:, comp_best:comp_best + 1], y_s_test))
        all_press = np.sum(PRESS, axis=0)
        RMSEP_comp_best = np.sqrt(all_press / y_s_test.shape[0]) 
        
        return RMSEP, RMSEP_t0, RMSEP_comp_best, y_trans_test[:, comp_best:comp_best + 1]

   
    def predict(self, T_trans_test, P_master_std, coefs_cal, y_s_test, y_m_cal):
        X_s_trans = np.dot(T_trans_test, P_master_std.T)  # 从仪器转移到主仪器的X             
        y_s_predict = np.dot(X_s_trans, coefs_cal) + y_m_cal.mean(axis=0)        
        PRESS = np.square(np.subtract(y_s_predict, y_s_test))
        all_press = np.sum(PRESS, axis=0)
        RMSEP = np.sqrt(all_press / y_s_test.shape[0])      
        return RMSEP, y_s_predict

    
    def transform(self, T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, b_s_list):

        n, m = np.shape(T_s_test)
        T_trans_test_norm = np.zeros((n, m))
        y_trans_test_norm = np.zeros((n, m))
        
        scaler_T = MinMaxScaler((0, 1)).fit(self.T_s_std)
        Ts_test_norm = scaler_T.transform(T_s_test)
#         print 'Ts_test_norm:', Ts_test_norm       
        scaler_y = MinMaxScaler((0, 1)).fit(self.y_s_pre)
        ys_test_pre_norm = scaler_y.transform(ys_test_pre)
#         print 'ys_test_pre_norm:', ys_test_pre_norm      
        for j in range(m):
            t_list = []
            y_list = [] 
            for i in range(n):
                t = Ts_test_norm[i, j]
                y = ys_test_pre_norm[i]
             
                t_trans = t * cos_x_list[j] - (y - b_s_list[j]) * sin_x_list[j]  #################################
                y_trans = t * sin_x_list[j] + (y - b_s_list[j]) * cos_x_list[j] + b_s_list[j]  #################################
                y_trans = y_trans + bia_list[j]
                
                t_list.append(t_trans)
                y_list.append(y_trans)
                
            T_trans_test_norm[:, j:j + 1] = np.array(t_list).reshape(-1, 1)
            y_trans_test_norm[:, j:j + 1] = np.array(y_list).reshape(-1, 1)
        
        scaler_T = MinMaxScaler((0, 1)).fit(self.T_m_std)
        T_trans_test = scaler_T.inverse_transform(T_trans_test_norm)
        
        scaler_y = MinMaxScaler((0, 1)).fit(self.y_m_pre)
        y_trans_test = scaler_y.inverse_transform(y_trans_test_norm)
        
        return T_trans_test, y_trans_test
    
    def transform_T0y(self, T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, b_s_list):

        n, m = np.shape(T_s_test)
        T_trans_test_norm = np.zeros((n, m))
                
        scaler_T = MinMaxScaler((0, 1)).fit(self.T_s_std)
        Ts_test_norm = scaler_T.transform(T_s_test)
 
        for j in range(m):
            t_list = []
            t0_list = [] 
            for i in range(n):
                t = Ts_test_norm[i, j]
                t0 = Ts_test_norm[i, 0]
                            
                t_trans = t * cos_x_list[j] - (t0 - b_s_list[j]) * sin_x_list[j]  #################################
                y_trans = t * sin_x_list[j] + (t0 - b_s_list[j]) * cos_x_list[j] + b_s_list[j]  #################################
                y_trans = y_trans + bia_list[j]
                
                t_list.append(t_trans)
                        
            T_trans_test_norm[:, j:j + 1] = np.array(t_list).reshape(-1, 1)
                 
        scaler_T = MinMaxScaler((0, 1)).fit(self.T_m_std)
        T_trans_test = scaler_T.inverse_transform(T_trans_test_norm)
        
               
        return T_trans_test

# if __name__ == '__main__':
# 
#     demo = fm.Dataset_Import(type=0, std_num=14, cal_size=0.8)
#     X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test = demo.Dataset_split()
#     W_m_cal, T_m_cal, P_m_cal, comp_best , coefs_cal = demo.Cal_model(max_comp=15, folds=10)
# 
#     T_m_std, T_s_std, y_m_pre, y_s_pre, T_s_test, ys_test_pre = fm.Pre_deal(X_m_std, X_s_std, W_m_cal, P_m_cal, coefs_cal, X_m_cal, y_m_cal, X_s_test)
#         
#     demo = Affine_trans(T_m_std, T_s_std, y_m_pre, y_s_pre, comp_best)
#     bia_list, sin_x_list, cos_x_list, x_list, b_s_list = demo.AT_train() 
#    
#     k, b = demo.AT_fit(T_s_std, y_s_pre, bia_list, cos_x_list, sin_x_list, b_s_list, y_s_std)      
#     RMSEP = demo.AT_pre(T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, y_s_test)
#                
#     print 'RMSEP:', RMSEP
#     sbc = SBC(X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test, n_folds=10, max_components=15)
#     bias_y, coefs_B_src, comp_best = sbc.transform()
#     RMSEP_sbc, y_tar_predict = sbc.predict(bias_y, coefs_B_src)  
#     print 'RMSEP_sbc',RMSEP_sbc           
#     RMSEP_sbc_list.append(RMSEP_sbc)
#     RMSEP_list.append(RMSEP)
      
 
 
      
if __name__ == '__main__':
    import time  
    from scipy.stats import wilcoxon   
    from compiler.ast import flatten  
    from PDS_910 import PDS
    from function_module import Dataset_KS_split_std
    
    RMSEP_sbc_list = []
    RMSEP_list = []
    RMSEP_to_list = []
    RMSEP_pds_list = []
    comp_best_list = []
    best_width_list = []
    
#     fname = loadmat('Pharmaceutical tablet')
#     print fname.keys()
#     D = fname
#     X_m_cal = D['calibrate_2']['data'][0][0]
#     X_s_cal = D['calibrate_1']['data'][0][0]
#     y_m_cal = D['calibrate_Y']['data'][0][0][:, 0:1]
#     y_s_cal = y_m_cal                
#     X_m_test = D['test_2']['data'][0][0]
#     X_s_test = D['test_1']['data'][0][0]
#     y_m_test = D['test_Y']['data'][0][0][:, 0:1]
#     y_s_test = y_m_test
#     print np.shape(X_m_cal), np.shape(y_m_cal)  # (155, 650) (155, 3)
#     print np.shape(X_m_test), np.shape(y_m_test)  # (460, 650) (460, 3)
         
#     for i in range(6, 7, 1):
#         print 'num:', i
    for j in range(20,30,1):
        print j
        demo = fm.Dataset_Import(type=13, std_num=2, cal_size=0.8, bool = 0)
        X_master , X_slave , y = demo.dataset_return()
        num = int(X_master.shape[0]*0.8)
        
        X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std = Dataset_KS_split_std(X_master , X_slave , y , num , std_num = j)
#         W_m_cal, T_m_cal, P_m_cal, comp_best , coefs_cal = demo.Cal_model(max_comp=15, folds=10)
        
    #         print np.shape(X_m_cal), np.shape(X_s_test)
        #         KS_master_std = KennardStone(X_m_cal, i)
        #         CalInd_master_std, ValInd_master_std = KS_master_std.KS()
        #         X_m_std = X_m_cal[CalInd_master_std]
        #         y_m_std = y_m_cal[CalInd_master_std]
        #         X_s_std = X_s_cal[CalInd_master_std]
        #         y_s_std = y_s_cal[CalInd_master_std]       
    #         pls_cal = Partial_LS(X_m_cal, y_m_cal, folds=10, max_comp=15)
    #         W_m_cal, T_m_cal, P_m_cal, comp_best , coefs_cal, RMSECV = pls_cal.pls_fit()
    #         print np.shape(X_m_cal), np.shape(X_s_test)
        
    #         print 'comp_best:', comp_best
#         T_m_std, T_s_std, y_m_pre, y_s_pre, T_s_test, ys_test_pre = fm.Pre_deal(X_m_cal, X_s_cal, W_m_cal, P_m_cal, coefs_cal, X_m_cal, y_m_cal, X_s_test)
#         
#         demo = Affine_trans(T_m_std, T_s_std, y_m_pre, y_s_pre, comp_best)
#         bia_list, sin_x_list, cos_x_list, x_list, b_s_list = demo.AT_train() 
#         #         print 'bs:', b_s_list   
#     #         k, b = demo.AT_fit(T_s_std, y_s_pre, bia_list, cos_x_list, sin_x_list, b_s_list, y_s_std)      
#         RMSEP , RMSEP_t0, RMSEP_comp_best, y_pre = demo.AT_pre(T_s_test, ys_test_pre, bia_list, cos_x_list, sin_x_list, y_s_test , b_s_list)
        
    #         print 'RMSEP/RMSEP_to/RMSEP_comp_best:', RMSEP, RMSEP_t0  , RMSEP_comp_best    
        
        
        
        
        sbc = SBC(X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test, n_folds=10, max_components=15)
        bias_y, coefs_B_src, comp_best = sbc.transform()
        RMSEP_sbc, y_sbc_predict = sbc.predict(bias_y, coefs_B_src , X_s_test ,y_s_test)    
        
        
        
        
    #         print 'y_test', y_s_test.tolist()
    #         print 'AT_pre=', y_pre.tolist()
    #         print 'SBC_pre=', y_sbc_predict.tolist()
           
        start = time.time()
        n_folds = 5
        max_components = 3  # 局部模型
        max_width = 16
        init_width = 3
        n_folds_cal = 5
        max_components_cal = 15
    ####################################################################### 
    #         demo = fm.Dataset_Import(type=3, std_num=26, cal_size=0.8)
    #         X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test, X_s_cal, y_s_cal = demo.Dataset_split()
    #         W_m_cal, T_m_cal, P_m_cal, comp_best , coefs_cal = demo.Cal_model(max_comp=15, folds=10)
    #######################################################################
        pds = PDS(X_m_cal, X_s_cal, y_m_cal, X_m_std, X_s_std, X_s_test, y_s_test, init_width, max_width)
        print n_folds , max_components , n_folds_cal , max_components_cal
        Trans_matrix_list = pds.transform3(n_folds, max_components)
        best_width, best_trans_matrix, coefs_B_cal, err_list, RMSECV_cal, comp_best_cal = pds.cv_window(Trans_matrix_list, n_folds_cal, max_components_cal)
        y_pds_pre, RMSEP_pds = pds.predict(best_trans_matrix, coefs_B_cal , X_s_test, y_s_test)
        print 'RMSEP_SBC:', RMSEP_sbc
#         print best_width
        print 'RMSEP_PDS:', RMSEP_pds
#     end = time.time()
#         print 'best_width:', best_width
# #         
#         print comp_best
#         print 'RMSEP/RMSEP_to:', RMSEP, RMSEP_t0
#         print np.shape(y_pre), np.shape(y_pds_pre)
    
#         print 'PDS_pre=', y_pds_pre.tolist()
#     y_at = flatten(np.subtract(y_pre, y_s_test).tolist())
#     y_sbc = flatten(np.subtract(y_sbc_predict, y_s_test).tolist())
#     y_pds = flatten(np.subtract(y_pds_pre, y_s_test).tolist())
    
    
#     t, p = wilcoxon(y_at, y_sbc)
#     t1, p1 = wilcoxon(y_at, y_pds)
#         print 'P_sbc,P_pds:', p, p1
#     print "times"
#     print 'RMSEP_SBC:', RMSEP_sbc
# #         print best_width
#     print 'RMSEP_PDS:', RMSEP_pds
#         print 'time:', end - start
#         RMSEP_list.append(RMSEP)
#         RMSEP_list.append(RMSEP_comp_best)
#     RMSEP_sbc_list.append(RMSEP_sbc)
#     RMSEP_pds_list.append(RMSEP_pds)
#     
#     print 'comp:', comp_best_list   
#     print 'af:', RMSEP_list
#     print 'af_to:', RMSEP_to_list
#     print 'sbc:', RMSEP_sbc_list
#     print 'pds:', RMSEP_pds_list
#     print 'width:', best_width_list
    
 
    
       
#     X = [5, 12, 19, 26, 33, 40, 47, 54, 61]
# #     X = [5, 8, 11, 14, 17, 20, 23, 26, 29]
# #     X = [8, 30, 52, 74, 96, 118, 140, 162, 184]
# #     X = [5, 23, 41, 59, 77, 95, 113, 131, 149]
#     X = np.array(X).reshape(-1, 1)
#     plt.plot(X, np.array(RMSEP_pds_list).reshape(-1, 1), '-o', color='blue')
#     plt.plot(X, np.array(RMSEP_sbc_list).reshape(-1, 1), '-o', color='red')
#   
#     plt.xticks(range(5, 64, 7))
#     plt.xlabel('number of standard sample')
#     plt.ylabel('RMSEP')
#     plt.legend(('pds_rmsep', 'SBC'), 'upper right')  # Affine_trans_rmsep
#     plt.show()
# #         
#     print RMSEP_sbc_list
#     print RMSEP_pds_list
#     print RMSEPstd_list
#     drawing.line_rmsep_stdNum(np.array(bias_y_list).reshape(-1, 1), np.array(RMSEP_list).reshape(-1, 1))

'''''' 
'''
if __name__ == '__main__':

#     import matplotlib.pyplot as plt
    fname = loadmat('NIRcorn.mat')
    print fname.keys()
    X_master = fname['m5spec']['data'][0][0]
    y = fname['cornprop'][:, 0:1]
    X_slave = fname['mp6spec']['data'][0][0]
#     (80, 700)    (80, 700)    (80, 1)
#     fname = loadmat('wheat_A_cal.mat')
#     print fname.keys()
#     X_master = fname['CalSetA3']
#     y = fname['protein']
#     X_slave = fname['CalSetA1']
     # (248, 741)      (248, 741)      (248, 1)
#     fname = loadmat('wheat_B_cal.mat')
#     print fname.keys()
#     X_master = fname['CalSetB2']
#     X_slave = fname['CalSetB3']
#     y = fname['protein']  
# #     
    print np.shape(X_master)    
    print np.shape(X_slave)    
    print np.shape(y)
   
    n_folds = 10
    max_components = 5
    
    pls_list = []
    psbct_list = []
    infor_list = []

    for i in range(5, 30, 3):
             
             
        print '\n\n           i,', (i)
        num = int(round(X_master.shape[0] * 0.8))
                    ##############        划分数据集
        KS_master = KennardStone(X_master, num)
        CalInd_master, ValInd_master = KS_master.KS()
        X_m_cal = X_master[CalInd_master]
        y_m_cal = y[CalInd_master]
        X_m_test = X_master[ValInd_master]
        y_m_test = y[ValInd_master]
             
        X_s_cal = X_slave[CalInd_master]
        X_s_test = X_slave[ValInd_master]
        y_s_cal = y[CalInd_master]
        y_s_test = y[ValInd_master]
             
             
        KS_master_std = KennardStone(X_m_cal, i)
        CalInd_master_std, ValInd_master_std = KS_master_std.KS()
        X_m_std = X_m_cal[CalInd_master_std]
        y_m_std = y_m_cal[CalInd_master_std]
        X_s_std = X_s_cal[CalInd_master_std]
        y_s_std = y_s_cal[CalInd_master_std]
        
        sbc = SBC(X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test, n_folds, max_components)
        bias_y, coefs_B_src, comp_best = sbc.transform()
        print bias_y
        pls_test = _NIPALS(comp_best)
        W_s_test, T_s_test, P_s_test, coef_s_test = pls_test.fit(X_s_test, y_s_test, comp_best)
        
#         T_trans_test = sbc.affine_predict(T_s_std, bia_list, sin_x_list, cos_x_list)
#         print 'T_trans_test', T_trans_test
#         print 'T_m_std', T_m_std
        T_trans_test = sbc.affine_predict(T_s_test, bia_list, sin_x_list, cos_x_list)
#         print 'T_trans_test', T_trans_test
        
        RMSEP, y_tar_predict = sbc.predict(bias_y, coefs_B_src)
        
        RMSEP1 = sbc.predict1(T_trans_test, P_m_std, coefs_B_src)
        
#         infor = []
#         infor.append(bia)
#         infor.append(x)
#         infor.append(comp_best)
# #         print 'infor', infor
#         infor_list.append(infor)
        pls_list.append(RMSEP)
        
        psbct_list.append(RMSEP1)
        
#         print 'comp_best_transfer:', comp_best_transfer
#         print 'comp_best_new:', comp_best_new
#         print 'PSBCT_RMSEP:', psbct_RMSEP
        
        
    corn_result_dict = {'PLS_RMSEP':pls_list, 'PSBCT_RMSEP':psbct_list}        
    corn_result = 'E:\\workspace\\algorithm\\CT-pls\\corn_m5_mp6.mat'
    savemat(corn_result, corn_result_dict, oned_as='row')           
        
    fname = loadmat('corn_m5_mp6')
    print fname.keys()
    pls_rmsep = fname['PLS_RMSEP']
    psbct_rmsep = fname['PSBCT_RMSEP']


#     drawing.line_rmsep_stdNum(pls_rmsep[9:18], psbct_rmsep[9:18], new_psbct_rmsep[9:18])
#     drawing.line_rmsep_stdNum(pls_rmsep[18:27], psbct_rmsep[18:27], new_psbct_rmsep[18:27])
#     drawing.line_rmsep_stdNum(pls_rmsep[27:36], psbct_rmsep[27:36], new_psbct_rmsep[27:36])
#     drawing.line_rmsep_stdNum(pls_rmsep[36:45], psbct_rmsep[36:45], new_psbct_rmsep[36:45])

                
    print '\n SBC_RMSEP  ,sbc_affine_RMSEP,        bia        ,        x        ,comp_best\n'  
    for i in range(9):

        print pls_rmsep[i], psbct_rmsep[i], '\n'  # , infor_list[i]
#         if (i + 1) % 9 == 0 :
#             print '\n\n\n\n'     
    drawing.line_rmsep_stdNum1(pls_rmsep[0:9], psbct_rmsep[0:9])
''' 
    
    

    


