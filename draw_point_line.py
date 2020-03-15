#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab.mio import loadmat
from PLS import Partial_LS
from function_module import *
from Affine_2018_7_16 import Affine_trans1

def deal_4_2(x , i , j):
    x_ = np.zeros((x.shape[0] , 2))
    x_[:,0] = x[:, i].ravel()
    x_[:,1] = x[:, j].ravel()
    return x_

if __name__ == "__main__":
    
    
    dataImport = Dataset_Import(type = 14, std_num = 8 , bool = 0)
    X_master , X_slave , y = dataImport.dataset_return()
    num = int(X_master.shape[0]*0.8)
    
    X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std = Dataset_KS_split_std(X_master , X_slave , y , num , std_num = 8)
    
    pls_demo = Partial_LS(X_m_cal,y_m_cal, folds=10 , max_comp=15)
    W_m_cal, T_m_cal, P_m_cal, comp_best, coefs_cal, RMSECV = pls_demo.pls2_fit()
#       
    ranges_x = [(9.5 , 11.5) , (3,4) , (7.5,10) , (62.5 , 66.5)]
    ranges_y = [(7 , 12.5) , (3,4.5), (6,10) , (62.5 , 68)]
    name=["A","B","C","D"]
    
#    
    plt.figure(dpi = 80)
    for i in range(4):
        for j in range(4):
            if(i == 0 and j == 1) or (i == 1 and j == 0) or (i==2 and j==3) or (i == 3 and j ==2):
                y_s_test_ = deal_4_2(y_s_test, i, j)
                y_s_cal_ = deal_4_2(y_s_cal,i, j)
                y_ = deal_4_2(y_m_cal, i, j)
                print "ok"
                coef = deal_4_2(coefs_cal, i, j)
                T_m_std, T_s_std, y_m_pre, y_s_pre, T_s_test, ys_test_pre = Pre_deal(X_m_cal, X_s_cal, W_m_cal, P_m_cal, coef, X_m_cal, y_, X_s_test)
                y0_m = y_m_pre[:, 0:1]
                y1_m = y_m_pre[:, 1:2]
                y0_s = y_s_pre[:, 0:1]
                y1_s = y_s_pre[:, 1:2]
                y0_s_test = ys_test_pre[:, 0:1]
                y1_s_test = ys_test_pre[:, 1:2]
#                 print y0_m , y1_m
                demo = Affine_trans1(y0_m, y0_s, y1_m, y1_s, comp_best)
                bia, sin_x, cos_x, x, b_s , k_m , b_m  = demo.AT_train()   
                
                RMSEP_y0, RMSEP_y1 , y0_pre = demo.AT_pre(y0_s_test, y1_s_test, bia, cos_x, sin_x, y_s_test_, b_s)
                RMSEC_y0, RMSEC_y1 , y_cal_pre = demo.AT_pre(y0_s, y1_s, bia, cos_x, sin_x, y_s_cal_, b_s)       
#                 
               
                plt.subplot(2,3,i + 1)
                plt.plot(ranges_x[i] , ranges_x[i] , color="black")
                plt.scatter(y_s_test[:,i], y0_pre[:,0], s = 10, c = "red" , label="corrected" , marker="x")
                plt.scatter(y_s_test[:,i], y0_s_test[:,0], s = 10, c = "green" , label="uncorrected")
                plt.legend(loc="upper left")
                plt.xlabel("reference values")
                plt.ylabel("predicted values")
                plt.title(name[i])
                # ���������᷶Χ
                plt.xlim(ranges_x[i])
                plt.ylim(ranges_y[i])
    
    
    
    
    
    
    
    
    
    
    dataImport = Dataset_Import(type = 13, std_num = 8 , bool = 0)
    X_master , X_slave , y = dataImport.dataset_return()
    num = int(X_master.shape[0]*0.8)
    
    X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std = Dataset_KS_split_std(X_master , X_slave , y , num , std_num = 8)
    
    pls_demo = Partial_LS(X_m_cal,y_m_cal, folds=10 , max_comp=15)
    W_m_cal, T_m_cal, P_m_cal, comp_best, coefs_cal, RMSECV = pls_demo.pls2_fit()
#       
#     ranges_x = [(9.5 , 11.5) , (3.0,4.0) , (7.5,10.1) , (62.5 , 66.7)]
#     ranges_y = [(9, 12.5) , (2.8,4.0), (7.5,12.3) , (60.5 , 66.7)]
#     name=["A","B","C","D"]
    
    ranges_x = [(350 , 400) , (15 , 25) , (150 , 225)]
    ranges_y = [(350 , 400) , (15 , 25) , (150 , 225)]
    name = ["D" , "E" , "F"]
    for i in range(1 , 3 , 1):
        for j in range(3):
            if(i == 1 and j == 2) or (i==2 and j==0):
                y_s_test_ = deal_4_2(y_s_test, i, j)
                y_s_cal_ = deal_4_2(y_s_cal,i, j)
                y_ = deal_4_2(y_m_cal, i, j)
                coef = deal_4_2(coefs_cal, i, j)
                T_m_std, T_s_std, y_m_pre, y_s_pre, T_s_test, ys_test_pre = Pre_deal(X_m_cal, X_s_cal, W_m_cal, P_m_cal, coef, X_m_cal, y_, X_s_test)
                y0_m = y_m_pre[:, 0:1]
                y1_m = y_m_pre[:, 1:2]
                y0_s = y_s_pre[:, 0:1]
                y1_s = y_s_pre[:, 1:2]
                y0_s_test = ys_test_pre[:, 0:1]
                y1_s_test = ys_test_pre[:, 1:2]
#                 print y0_m , y1_m
                demo = Affine_trans1(y0_m, y0_s, y1_m, y1_s, comp_best)
                bia, sin_x, cos_x, x, b_s , k_m , b_m  = demo.AT_train()   
                
                RMSEP_y0, RMSEP_y1 , y0_pre = demo.AT_pre(y0_s_test, y1_s_test, bia, cos_x, sin_x, y_s_test_, b_s)
                RMSEC_y0, RMSEC_y1 , y_cal_pre = demo.AT_pre(y0_s, y1_s, bia, cos_x, sin_x, y_s_cal_, b_s)       
#                 
               
                plt.subplot(2,3,i+4)
                plt.plot(ranges_x[i] , ranges_x[i] , color="black")
                plt.scatter(y_s_test[:,i], y0_pre[:,0], s = 10, c = "red" , label="corrected" , marker="x")
                plt.scatter(y_s_test[:,i], y0_s_test[:,0], s = 10, c = "green" , label="uncorrected")
                plt.legend(loc="upper left")
                plt.xlabel("reference values")
                plt.ylabel("predicted values")
                plt.title(name[i])
                # ���������᷶Χ
                plt.xlim(ranges_x[i])
                plt.ylim(ranges_y[i])
    
    
    plt.show()

    