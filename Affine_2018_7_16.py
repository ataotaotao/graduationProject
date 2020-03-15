# -*- coding: utf-8 -*-
'''
Created on 2018年7月27日
针对PLS2，将两列预测值y之间，进行仿射变换
@author: Administrator
'''
from scipy.io.matlab.mio import loadmat
from PLS import Partial_LS
from function_module import Dataset_KS_split
import numpy as np
import function_module as fm
from pls_demo import PlsDemo
from NIPALS import _NIPALS
import math
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn import cross_validation 
from sklearn.decomposition import PCA
from sklearn import linear_model
from KS_algorithm import KennardStone

class Affine_trans1():
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
#         print Tm_std_norm.reshape(-1, 1).shape ,"999999999996666666" , y_m_pre_norm.reshape(-1, 1).shape
        clf.fit (Tm_std_norm.reshape(-1, 1), y_m_pre_norm.reshape(-1, 1))
        k_m_std = clf.coef_
        b_m_std = clf.intercept_
        
        clf = linear_model.LinearRegression(fit_intercept=True)
        clf.fit (Ts_std_norm.reshape(-1, 1), y_s_pre_norm.reshape(-1, 1))
        k_s_std = clf.coef_
        b_s_std = clf.intercept_   
             
#         print ' master , k,b', k_m_std, b_m_std
#         print k_s_std, b_s_std
          
        T = np.array([min_value, max_value]).reshape(-1, 1)
        y_m_start = k_m_std * T[0] + b_m_std
        y_m_end = k_m_std * T[1] + b_m_std
        y_s_start = k_s_std * T[0] + b_s_std
        y_s_end = k_s_std * T[1] + b_s_std
        y_m = np.array([y_m_start, y_m_end]).reshape(-1, 1)
        y_s = np.array([y_s_start, y_s_end]).reshape(-1, 1)
        
#         if draw == True:
#             drawing.line_fit(T, y_m, y_s, Tm_std_norm, Ts_std_norm, y_m_pre_norm, y_s_pre_norm)
            
        bia = b_m_std - b_s_std
    
            # x1,y1Ϊ��ʵֱ�ߵ�������ʽ            x2,y2ΪԤ��ֱ�ߵ�������ʽ
            
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
        return bia, sin_x, cos_x, x, b_s_std , k_m_std , b_m_std

    def AT_train(self):
   
        y_m_pre_norm = fm.Norm(self.y_m_pre)
        y_s_pre_norm = fm.Norm(self.y_s_pre)        
        Tm_std_norm = fm.Norm(self.T_m_std)
        Ts_std_norm = fm.Norm(self.T_s_std)
        bia, sin_x, cos_x, x, b_s , k_m_std , b_m_std = self.affine_trans(Tm_std_norm, Ts_std_norm, y_m_pre_norm, y_s_pre_norm, draw=False)           
   
        return bia, sin_x, cos_x, x, b_s , k_m_std , b_m_std

    
    def AT_pre(self, y0_s_test, y1_s_test, bia_list, cos_x_list, sin_x_list, y_s_test, b_s_list):
     
        y0_trans_test, y1_trans_test = self.transform(y0_s_test, y1_s_test, bia_list, cos_x_list, sin_x_list, b_s_list)
#         print np.shape(y0_trans_test), np.shape(y1_trans_test)
        
        PRESS = np.square(np.subtract(y0_trans_test, y_s_test[:, 0:1]))
        all_press = np.sum(PRESS, axis=0)
        RMSEP_y0 = np.sqrt(all_press / y_s_test.shape[0]) 
         
        PRESS = np.square(np.subtract(y1_trans_test, y_s_test[:, 1:2]))
        all_press = np.sum(PRESS, axis=0)
        RMSEP_y1 = np.sqrt(all_press / y_s_test.shape[0]) 
              
        return RMSEP_y0, RMSEP_y1 , y0_trans_test
    
    def AT_train_pre(self , y0_s, y1_s, bia, cos_x, sin_x, y_s_cal, b_s):
        y0_s_cal_pre, y1_s_cal_pre = self.transform(y0_s, y1_s, bia, cos_x, sin_x, b_s)
#         print y0_s_cal_pre.shape , y_s_cal.shape
        PRESS = np.square(np.subtract(y0_s_cal_pre, y_s_cal[:, 0:1]))
        all_press = np.sum(PRESS, axis=0)
        RMSEC_y0 = np.sqrt(all_press / y_s_cal.shape[0]) 
        
        PRESS = np.square(np.subtract(y1_s_cal_pre, y_s_cal[:, 1:2]))
        all_press = np.sum(PRESS, axis=0)
        RMSEC_y1 = np.sqrt(all_press / y_s_cal.shape[0]) 
        
        return RMSEC_y0 , RMSEC_y1 ,y0_s_cal_pre
    def AT_pre_no_trans(self, y0, y1 , y_s_test):
        PRESS = np.square(np.subtract(y0, y_s_test[:, 0:1]))
        all_press = np.sum(PRESS, axis=0)
        RMSEP_y0 = np.sqrt(all_press / y_s_test.shape[0]) 
         
        PRESS = np.square(np.subtract(y1, y_s_test[:, 1:2]))
        all_press = np.sum(PRESS, axis=0)
        RMSEP_y1 = np.sqrt(all_press / y_s_test.shape[0]) 
              
        return RMSEP_y0, RMSEP_y1 , y0
        
        
        
        
        
    def transform(self, T_s_test, ys_test_pre, bia, cos_x, sin_x, b_s):

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
                
                t_trans = t * cos_x - (y - b_s) * sin_x  #################################
                y_trans = t * sin_x + (y - b_s) * cos_x + b_s  #################################
                y_trans = y_trans + bia
                
                t_list.append(t_trans)
                y_list.append(y_trans)
                
            T_trans_test_norm[:, j:j + 1] = np.array(t_list).reshape(-1, 1)
            y_trans_test_norm[:, j:j + 1] = np.array(y_list).reshape(-1, 1)
        
        scaler_T = MinMaxScaler((0, 1)).fit(self.T_m_std)
        T_trans_test = scaler_T.inverse_transform(T_trans_test_norm)
        
        scaler_y = MinMaxScaler((0, 1)).fit(self.y_m_pre)
        y_trans_test = scaler_y.inverse_transform(y_trans_test_norm)
        
        return T_trans_test, y_trans_test
'''
if __name__ == '__main__':
    fname = loadmat('NIRcorn.mat')
    print fname.keys()
    X_master = fname['m5spec']['data'][0][0]
    y = fname['cornprop'][:, 0:2]
    X_slave = fname['mp6spec']['data'][0][0]
    
    fname = loadmat('Pharmaceutical tablet')
    print fname.keys()
    D = fname
    
#     X_m_cal = D['calibrate_2']['data'][0][0]
#     X_s_cal = D['calibrate_1']['data'][0][0]
#     y_m_cal = D['calibrate_Y']['data'][0][0][:, 0:1]
#     y_s_cal = y_m_cal                
#     X_m_test = D['test_2']['data'][0][0]
#     X_s_test = D['test_1']['data'][0][0]
#     y_m_test = D['test_Y']['data'][0][0][:, 0:1]
#     y_s_test = y_m_test 
        
       
    X_master = D['calibrate_2']['data'][0][0]
    X_slave = D['calibrate_1']['data'][0][0]
    y = D['calibrate_Y']['data'][0][0][:, 2:3]    
    num = int (X_master.shape[0] * 0.8)
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
    
    sbc_list = [] 
    for std_num in range(5, 40, 3):  
        print 'std_num:', std_num 
        KS_master_std = KennardStone(X_m_cal, std_num)
        CalInd_master_std, ValInd_master_std = KS_master_std.KS()
        X_m_std = X_m_cal[CalInd_master_std]
        y_m_std = y_m_cal[CalInd_master_std]
        X_s_std = X_s_cal[CalInd_master_std]
        y_s_std = y_s_cal[CalInd_master_std]
    
        sbc = SBC(X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test, n_folds=10, max_components=15)
        bias_y, coefs_B_src, comp_best = sbc.transform()
        RMSEP_sbc, y_sbc_predict = sbc.predict(bias_y, coefs_B_src)        
        print 'RMSEP_SBC:', RMSEP_sbc  
        sbc_list.append(RMSEP_sbc)
    
    print sbc_list
 

'''
    #计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean
#计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)    #计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

if __name__ == '__main__':
    from function_module import SNV
#     fname = loadmat("Pharmaceutical tablet.mat")
#     X_master = fname['test_1']['data'][0][0]
#     X_slave = fname['test_2']['data'][0][0]
#     y = fname['test_Y']['data'][0][0][:, 0:3]
     
     
#     X_master = SNV(X_master)
#     X_slave = SNV(X_slave)
    
   
    
    
    fname = loadmat('NIRcorn.mat')
    
    print fname.keys()
    X_master = fname['mp5spec']['data'][0][0]
    X_slave = fname['mp6spec']['data'][0][0]
    y = fname['cornprop'][:, 0:4]
    
    X_master = SNV(X_master)
    X_slave = SNV(X_slave)
    
    
    
    
#     fname = loadmat('Pharmaceutical tablet')
#     print fname.keys()
#     D = fname
#     X_master = D['test_2']['data'][0][0]
#     X_slave = D['test_1']['data'][0][0]
#     y = D['test_Y']['data'][0][0][:, 0:2]          
    num = int (X_master.shape[0] * 0.8)
    print num
    X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test = Dataset_KS_split(X_master, X_slave, y, num)
    
#     X_m_cal = D['calibrate_2']['data'][0][0]
#     X_s_cal = D['calibrate_1']['data'][0][0]
#     y_m_cal = D['calibrate_Y']['data'][0][0][:, 0:2]
#     y_s_cal = y_m_cal                
#     X_m_test = D['test_2']['data'][0][0]
#     X_s_test = D['test_1']['data'][0][0]
#     y_m_test = D['test_Y']['data'][0][0][:, 0:2]
#     y_s_test = y_m_test 
    
    pls_demo = Partial_LS(X_m_cal, y_m_cal, folds=10 , max_comp=15)
    W_m_cal, T_m_cal, P_m_cal, comp_best, coefs_cal, RMSECV = pls_demo.pls2_fit()
#     print coefs_cal
    b1 = coefs_cal[:,0]
    b2 = coefs_cal[:,1]
    b3 = coefs_cal[:,2]
    b4 = coefs_cal[:,3]
    
    print calcPearson(b3, b4)
    
    
    
    
#     repeat_num = y_m_cal.shape[1]
#     for i in range(repeat_num):
#         for j in range(repeat_num):
#             if i == j :
#                 continue
#             y_ = np.zeros((y_m_cal.shape[0] , 2))
#             y_[:,0] = y_m_cal[:,i]
#             ######## 问题
#             y_[:,1] = y_m_cal[:,j]
#             coef = np.zeros((coefs_cal.shape[0] , 2))
#             coef[:,0] = coefs_cal[:,i]
#             ######## 问题
#             coef[:,1] = coefs_cal[:,j]
#             y_s_test_ = np.zeros((y_s_test.shape[0] , 2))
#             y_s_test_[:,0] = y_s_test[:,i]
#             ######## 问题
#             y_s_test_[:,1] = y_s_test[:,j]
#             T_m_std, T_s_std, y_m_pre, y_s_pre, T_s_test, ys_test_pre = fm.Pre_deal(X_m_cal, X_s_cal, W_m_cal, P_m_cal, coef, X_m_cal, y_, X_s_test)
#             y0_m = y_m_pre[:, 0:1]
#             y1_m = y_m_pre[:, 1:2]
#             y0_s = y_s_pre[:, 0:1]
#             y1_s = y_s_pre[:, 1:2]
#             y0_s_test = ys_test_pre[:, 0:1]
#             y1_s_test = ys_test_pre[:, 1:2]
#             
#             demo = Affine_trans1(y0_m, y0_s, y1_m, y1_s, comp_best)
#             bia, sin_x, cos_x, x, b_s = demo.AT_train()   
#         #     print bia, sin_x, cos_x, b_s
#             RMSEP_y0, RMSEP_y1 , y0_pre = demo.AT_pre(y0_s_test, y1_s_test, bia, cos_x, sin_x, y_s_test_, b_s)       
#             print 'RMSEP_y0/RMSEP_y1:', RMSEP_y0, RMSEP_y1
    
    
    
    
            














