# -*- coding: utf-8 -*-
'''
Created on 2018年4月16日
@author: Administrator
'''
import numpy as np

from PLS import Partial_LS
from KS_algorithm import KennardStone
from sklearn import linear_model
import math
from sklearn import cross_validation
from scipy.io.matlab.mio import loadmat, savemat
from scipy.stats.morestats import wilcoxon

class Dataset_Import():
    def __init__(self, type=0, std_num=5, cal_size=0.8 , bool = 0):
        self.std_num = std_num
        self.cal_size = cal_size
        
        fname1 = loadmat('NIRcorn.mat')
        fname = loadmat('wheat_A_cal.mat')
        fname2 = loadmat('wheat_B_cal.mat')
        fname3 = loadmat('Pharmaceutical tablet.mat')
        
        if type < 4:
#             print 'cornprop--y:', type
            self.X_master = fname1['mp5spec']['data'][0][0]
            self.y = fname1['cornprop'][:,type:type + 1]
            self.X_slave = fname1['mp6spec']['data'][0][0]
            
            
        if type == 4:
            print 'B1--B2'
            self.X_master = fname2['CalSetB1']
            self.X_slave = fname2['CalSetB2']
            self.y = fname2['protein']
        if type == 5:
            print 'B1--B3'
            self.X_master = fname2['CalSetB1']
            self.X_slave = fname2['CalSetB3']
            self.y = fname2['protein']
        if type == 6:
            print 'B2--B3'
            self.X_master = fname2['CalSetB2']
            self.X_slave = fname2['CalSetB3']
            self.y = fname2['protein']
        if type == 7:   
            print 'A2--A1'
            self.X_master = fname['CalSetA2']
            self.y = fname['protein']
            self.X_slave = fname['CalSetA1']
        if type == 8:  
            print 'A3--A1' 
            self.X_master = fname['CalSetA3']
            self.y = fname['protein']
            self.X_slave = fname['CalSetA1']
        if type == 9:  
            print 'A3--A2' 
            self.X_master = fname['CalSetA3']
            self.y = fname['protein']
            self.X_slave = fname['CalSetA2']
        if type == 10:   
            print 'tablet---0'
            self.X_master = fname3['validate_1']['data'][0][0]
            self.X_slave = fname3['validate_2']['data'][0][0]
            self.y = fname3['validate_Y']['data'][0][0][:, 0:1]
        if type == 11:  
            print 'tablet---1' 
            self.X_master = fname3['test_1']['data'][0][0]
            self.X_slave = fname3['test_2']['data'][0][0]
            self.y = fname3['test_Y']['data'][0][0][:, 1:2]
        if type == 12:  
            print 'tablet---2' 
            self.X_master = fname3['test_1']['data'][0][0]
            self.X_slave = fname3['test_2']['data'][0][0]
            self.y = fname3['test_Y']['data'][0][0][:, 2:3]
        if type == 13:  
            print 'tablet---all' 
            self.X_master = fname3['test_1']['data'][0][0]
            self.X_slave = fname3['test_2']['data'][0][0]
            self.y = fname3['test_Y']['data'][0][0][:, 0:3]
        if type == 14:
            self.X_master = fname1['m5spec']['data'][0][0]
            self.y = fname1['cornprop'][:,0:4]
            self.X_slave = fname1['mp6spec']['data'][0][0]
        if bool == 1 :
            print "snv_deal"
            self.X_master = SNV(self.X_master)
            self.X_slave = SNV(self.X_slave)
#         drawing.spectrum(self.X_master)
#         drawing.spectrum(self.X_slave)
#         drawing.spectrum(np.subtract(self.X_master, self.X_slave))
    def dataset_return(self):
        return self.X_master , self.X_slave , self.y 
    def Dataset_split(self):
        
        num = int(round(self.X_master.shape[0] * self.cal_size))
                    ##############        划分数据集
        KS_master = KennardStone(self.X_master, num)
        CalInd_master, ValInd_master = KS_master.KS()
        X_m_cal = self.X_master[CalInd_master]
        y_m_cal = self.y[CalInd_master]
        X_m_test = self.X_master[ValInd_master]
        y_m_test = self.y[ValInd_master]
#         print CalInd_master , ValInd_master
        X_s_cal = self.X_slave[CalInd_master]
        X_s_test = self.X_slave[ValInd_master]
        y_s_cal = self.y[CalInd_master]
        y_s_test = self.y[ValInd_master]
        
        KS_master_std = KennardStone(X_m_cal, self.std_num)
        CalInd_master_std, ValInd_master_std = KS_master_std.KS()
        X_m_std = X_m_cal[CalInd_master_std]
        y_m_std = y_m_cal[CalInd_master_std]
        X_s_std = X_s_cal[CalInd_master_std]
        y_s_std = y_s_cal[CalInd_master_std]
             
        return X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test, X_s_cal, y_s_cal, X_m_test
    
    def Cal_model(self, max_comp, folds):
        
        X_m_cal, y_m_cal, X_m_std, y_m_std, X_s_std, y_s_std, X_s_test, y_s_test, X_s_cal, y_s_cal, X_m_test = self.Dataset_split()        
        
        pls_cal = Partial_LS(X_m_cal, y_m_cal, folds, max_comp)
        W_m_cal, T_m_cal, P_m_cal, comp_best , coefs_cal, RMSECV = pls_cal.pls2_fit()

        return W_m_cal, T_m_cal, P_m_cal, comp_best , coefs_cal
    
    
def affine_trans(T_m_std, T_s_std, y_m_pre, y_s_pre, draw=False):
#     print 'T_m_std', T_m_std, T_s_std
#     print 'y', y_m_pre, y_s_pre
    Max_m = np.max(T_m_std, axis=0)
    Max_s = np.max(T_s_std, axis=0)
    Min_m = np.min(T_m_std, axis=0)
    Min_s = np.min(T_s_std, axis=0)
               
    if Max_m > Max_s:
        max_value = Max_m
    else:
        max_value = Max_s
       
    if Min_m > Min_s:
        min_value = Min_s
    else:
        min_value = Min_m
        
    clf = linear_model.LinearRegression(fit_intercept=True)
    clf.fit (T_m_std.reshape(-1, 1), y_m_pre.reshape(-1, 1))
    k_m_std = clf.coef_
    b_m_std = clf.intercept_
  
    clf = linear_model.LinearRegression(fit_intercept=True)
    clf.fit (T_s_std.reshape(-1, 1), y_s_pre.reshape(-1, 1))
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
        drawing.line_fit(T, y_m, y_s, T_m_std, T_s_std, y_m_pre, y_s_pre)
        
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
    return bia, sin_x, cos_x, x


def Norm(data):  # 归一化
  
    Max = np.max(data)
    Min = np.min(data)
#     print 'Max,min', Max, Min
    n, m = np.shape(data)
    data_norm = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            data_norm[i, j] = (data[i, j] - Min) / (Max - Min)
            
    return data_norm
  
    
def compute_score(X, W, P):
    # 67 700
        n, m = np.shape(X)
        n_w, m_w = np.shape(W)
        T = np.ones((n, m_w))
        for i in range(m_w): 
            t = np.dot(X, W[:, i])
            t = np.mat(t).T
            p = np.mat(P[:, i])
            X = np.subtract(X, np.dot(t, p))
            T[:, i] = t.ravel()
            
        return T    

def Matrix_abs(X):
    n, m = np.shape(X)
    X_abs = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            X_abs[i, j] = np.abs(X[i, j])
    
    return X_abs
#             x_m_cal, x_s_cal, weight_W, loading_P, coefs_B, x_m_cal, ytrain_pre, x_s_test
def Pre_deal(X_m_std, X_s_std, W_m_cal, P_m_cal, coefs_cal, X_m_cal, y_m_cal, X_s_test):
    
    X_m_std_cen = np.subtract(X_m_std, np.mean(X_m_cal, axis=0))
    X_s_std_cen = np.subtract(X_s_std, np.mean(X_m_cal, axis=0))
   
    
    T_m_std = compute_score(X_m_std_cen, W_m_cal, P_m_cal)
    T_s_std = compute_score(X_s_std_cen, W_m_cal, P_m_cal)
    
    ######### PLS2使用的只有这两个
#     print X_m_std_cen.shape , coefs_cal.shape , y_m_cal.shape 
    y_m_pre = np.dot(X_m_std_cen, coefs_cal) + np.mean(y_m_cal, axis=0)
    # 这个就是cal训练的误差
    y_s_pre = np.dot(X_s_std_cen, coefs_cal) + np.mean(y_m_cal, axis=0)
    
    
    
    
    X_s_test_cen = np.subtract(X_s_test, np.mean(X_m_cal, axis=0))
    T_s_test = compute_score(X_s_test_cen, W_m_cal, P_m_cal)
    X_s_test_cen = np.subtract(X_s_test, np.mean(X_m_cal, axis=0))
    ys_test_pre = np.dot(X_s_test_cen, coefs_cal) + np.mean(y_m_cal, axis=0)
    

    return T_m_std, T_s_std, y_m_pre, y_s_pre, T_s_test, ys_test_pre

def trans_all_infor(T_m_std, T_s_std, y_m_pre_norm, y_s_pre_norm, comp_best):
    bia_list = []
    sin_x_list = []
    cos_x_list = []
    x_list = []
    for i in range(comp_best):
        Tm_std_norm = Norm(T_m_std[:, i:i + 1])
        Ts_std_norm = Norm(T_s_std[:, i:i + 1])
#         print T_m_std[:, i:i + 1]
#         print 'Tm_std_norm:', Tm_std_norm
#         print y_m_pre_norm
        bia, sin_x, cos_x, x = affine_trans(Tm_std_norm, Ts_std_norm, y_m_pre_norm, y_s_pre_norm, draw=False)
        bia_list.append(bia)
        sin_x_list.append(sin_x)
        cos_x_list.append(cos_x)
        x_list.append(x)
        
    return bia_list, sin_x_list, cos_x_list, x_list

def cv_split_data(X, y, folds):  # 划分训练集与测试集
    
    n = X.shape[0]
    
    kf = cross_validation.KFold(n, folds)
 
    x_train = []
    y_train = []
    x_test = [] 
    y_test = []
    for train_index, test_index in kf:
        xtr, ytr = X[train_index], y[train_index]
        xte, yte = X[test_index], y[test_index]
        x_train.append(xtr)
        y_train.append(ytr)
        x_test.append(xte)
        y_test.append(yte)
        
    return x_train, x_test, y_train, y_test       

def Dataset_KS_split(X_master, X_slave, y, num):
    KS_master = KennardStone(X_master, num)
    CalInd_master, ValInd_master = KS_master.KS()
#     print CalInd_master , ValInd_master
    X_m_cal = X_master[CalInd_master]
    y_m_cal = y[CalInd_master]
    X_m_test = X_master[ValInd_master]
    y_m_test = y[ValInd_master]
    
    X_s_cal = X_slave[CalInd_master]
    X_s_test = X_slave[ValInd_master]
    y_s_cal = y[CalInd_master]
    y_s_test = y[ValInd_master]             
    
    return X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test
    
def Dataset_KS_split_std(X_master, X_slave, y, num , std_num = 8):
    KS_master = KennardStone(X_master, num)
    CalInd_master, ValInd_master = KS_master.KS()
#     print CalInd_master , ValInd_master
    X_m_cal = X_master[CalInd_master]
    y_m_cal = y[CalInd_master]
    X_m_test = X_master[ValInd_master]
    y_m_test = y[ValInd_master]
    
    X_s_cal = X_slave[CalInd_master]
    X_s_test = X_slave[ValInd_master]
    y_s_cal = y[CalInd_master]
    y_s_test = y[ValInd_master]  
    
    KS_master_std = KennardStone(X_m_cal, std_num)
    CalInd_master_std, ValInd_master_std = KS_master_std.KS()
#     print CalInd_master_std , ValInd_master
    X_m_std = X_m_cal[CalInd_master_std]
    y_m_std = y_m_cal[CalInd_master_std]
    X_s_std = X_s_cal[CalInd_master_std]
    y_s_std = y_s_cal[CalInd_master_std]
    
    return X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std
def Data_KS_split(X_master, X_slave, y, num):
    KS_master = KennardStone(X_slave, num)
    CalInd_master, ValInd_master = KS_master.KS()
    X_m_cal = X_master[CalInd_master]
    y_cal = y[CalInd_master]
    X_m_test = X_master[ValInd_master]
    y_test = y[ValInd_master]
             
    X_s_cal = X_slave[CalInd_master]
    X_s_test = X_slave[ValInd_master]         
    
    return X_m_cal, y_cal, X_s_cal, X_s_test, X_m_test, y_test
    
    
def SNV(x):
    
    x_mean_ = np.mean(x , axis = 1)
    x_cov = np.subtract(x , x_mean_.reshape(-1,1))  / np.std( x , axis=1).reshape(-1,1)
    return x_cov

def evaluation(RMSEP , RMSEP_other , y , y_predict , y_other_predict):
    h = (1 - RMSEP / RMSEP_other) * 100
#     print y_predict.shape,y.shape
    y_new = np.subtract(y_predict , y).ravel()
    
    y_old = np.subtract(y_other_predict , y).ravel()
    p = wilcoxon(y_new , y_old)
#     print p[1] , 666
    return p[1] , h
    
def getmin(array):
    index = 0
    mins = array[0]
    print mins , len(array)
    for i in range(len(array)):
        if array[i]<mins:
            mins = array[i]
            index = i
#     多了一个recalibration ， 所以直接用就好了。
    return index+1 , mins


    
    
    
    
    
    
    

