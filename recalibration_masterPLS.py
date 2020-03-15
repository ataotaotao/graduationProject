#-*-coding:utf-8-*-
'''
Created on 2018��10��20��

@author: DELL
'''

from sklearn import cross_validation 
from sklearn.cross_validation import train_test_split
from scipy import stats
import sys
from idlelib.ReplaceDialog import replace

print(sys.path)
from SBC_2018_4_14 import SBC
from PDS_910 import PDS
from MSC_831 import MSC
from ccact import CCACT
from Affine_2018_7_16 import Affine_trans1

from PLS import Partial_LS
from excel_format_2 import Excel_format
from function_module import *
from demo import datasetProcess
from KS_algorithm import KennardStone
from tca_param_choose import tca_param_choose
from TCA import TCA

import numpy as np
from scipy import linalg
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.io import loadmat,savemat
from sklearn.cross_validation import train_test_split
from openpyxl import load_workbook
from openpyxl.styles import Font, colors, Alignment


def deal_4_2(x , i , j):
    x_ = np.zeros((x.shape[0] , 2))
    x_[:,0] = x[:, i].ravel()
    x_[:,1] = x[:, j].ravel()
    return x_
def deal_4_1(x , i):
    x_ = np.zeros((x.shape[0] , 1))
    x_[:,0] = x[:, i].ravel()
    return x_

class DEMO():
    def __init__(self,X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, 
                 X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std,max_folds,
                 max_components , method_list , 
                 RMSEC_list,RMSEP_list,ycal_predict_list,y_predict_list
                  ,comp_best_list , i):
        
        self.X_m_cal = X_m_cal 
        self.y_m_cal = y_m_cal
        self.X_s_cal = X_s_cal
        self.y_s_cal = y_s_cal
        self.X_s_test = X_s_test
        self.y_s_test = y_s_test
        self.X_m_test = X_m_test
        self.y_m_test = y_m_test
        self.X_m_std = X_m_std
        self.y_m_std = y_m_std
        self.y_m_std = y_m_std
        self.X_s_std = X_s_std
        self.y_s_std = y_s_std
        self.max_folds=max_folds
        self.max_components=max_components
        self.method_list = method_list
        self.comp_best_list = comp_best_list
        self.RMSEC_list = RMSEC_list
        self.RMSEP_list = RMSEP_list
        self.ycal_predict_list = ycal_predict_list
        self.y_predict_list = y_predict_list
        self.i = i
        
    def Recalibration(self):
        y_s_cal = deal_4_1(self.y_s_cal, self.i)
        y_s_test = deal_4_1(self.y_s_test, self.i)
#         print self.y_m_cal.shape , self.y_m_std.shape , self.y_m_test.shape , self.y_s_cal.shape
        pls=Partial_LS(self.X_s_cal,y_s_cal,self.max_folds,self.max_components)  
        W,T,P,comp_best,coefs_B,RMSECV=pls.pls_fit()
        RMSEC,RMSEP,yte_predict=pls.pls_pre(self.X_s_test,y_s_test,coefs_B)
        cal_RMSEC,cal_RMSEP ,ycal_predict=pls.pls_pre(self.X_s_cal,y_s_cal,coefs_B)
        
        return comp_best,RMSECV,RMSEP,yte_predict,RMSEC,ycal_predict
        
    def pls(self,n_folds):
        y_s_std = deal_4_1(self.y_s_std, self.i)
        y_s_cal = deal_4_1(self.y_s_cal, self.i)
        y_s_test = deal_4_1(self.y_s_test, self.i)
        pls=Partial_LS(self.x_s_std,y_s_std,n_folds,self.max_components)  
        W,T,P,comp_best,coefs_B,RMSECV=pls.pls_fit()
        RMSEC,RMSEP,yte_predict=pls.pls_pre(self.x_s_test,y_s_test,coefs_B)
        cal_RMSEC,cal_RMSEP,ycal_predict=pls.pls_pre(self.x_s_cal,y_s_cal,coefs_B)

        
        return comp_best,RMSECV,RMSEP,yte_predict,RMSEC,ycal_predict
    
    def TCA(self , num , i , snv = 0):
        
#################
        DP_demo = datasetProcess(dataType=i , bool = snv)
        X_master, X_slave, y = DP_demo.datasetImport()
        X_master_cal, X_master_test, X_slave_cal, X_slave_test, y_cal, y_test = DP_demo.datasetSplit_KS(X_master, X_slave, y, num)
        
        X_m_train, X_m_val, y_m_train, y_m_val = train_test_split(X_master_cal, y_cal, test_size=0.5, random_state=0)
        KS_demo = KennardStone(X_slave, num)
        CalInd, ValInd = KS_demo.KS()  
        
        X_s = X_slave[CalInd]  # �������б�ǩ������
        X_s_o = X_slave[ValInd]  # ������û�б�ǩ������
        y_s = y[CalInd]  # �����������ı�ǩ
        
        # ѡ�����
        TPC_demo = tca_param_choose(X_m_train, y_m_train, X_m_val, y_m_val, X_s, y_s, X_s_o)
        m_op, k_op = TPC_demo.choose_Param(15)
        print m_op , "m_op"
        
        my_tca = TCA(dim=m_op)
        
        T_m_cal, T_s_o, T_slave_test, T_s = my_tca.fit_transform(X_master_cal, X_slave_cal, X_slave_cal, X_s)
        T = np.vstack((T_m_cal, T_s))
        y = np.vstack((y_cal, y_s))
        k1 = np.linalg.lstsq(T, y)[0]
#         print np.shape(k1), np.shape(T_slave_test)
        y_cal_pre = np.dot(T_slave_test, k1)
        RMSEC = np.sqrt(np.sum(np.square(np.subtract(y_cal, y_cal_pre)), axis=0) / y_cal_pre.shape[0])
#         print RMSEC , "RMSEC"
        
        T_m_cal_, T_s_o, T_slave_test_, T_s_ = my_tca.fit_transform(X_master_cal, X_slave_cal, X_slave_test, X_s)
        T_ = np.vstack((T_m_cal_, T_s_))
        y_ = np.vstack((y_cal, y_s))
        k1_ = np.linalg.lstsq(T_, y_)[0]
#         print np.shape(k1), np.shape(T_slave_test_)
        y_test_pre_ = np.dot(T_slave_test_, k1_)
        RMSEP = np.sqrt(np.sum(np.square(np.subtract(y_test, y_test_pre_)), axis=0) / y_test_pre_.shape[0])
        
        return m_op,RMSEP,y_test_pre_,RMSEC,y_cal_pre
    def sbc(self):
        y_m_cal = deal_4_1(self.y_m_cal, self.i)
        y_m_std = deal_4_1(self.y_m_std, self.i)
        y_s_std = deal_4_1(self.y_s_std, self.i)
        y_s_test = deal_4_1(self.y_s_test, self.i)
        y_s_cal = deal_4_1(self.y_s_cal, self.i)
        sbc=SBC(self.X_m_cal,y_m_cal,self.X_m_std,y_m_std,self.X_s_std,y_s_std,self.X_s_test,y_s_test,self.max_folds,self.max_components)
        bias_y,coefs_B_m,comp_best_m=sbc.transform()
        RMSEP,yte_predict=sbc.predict( bias_y,coefs_B_m,self.X_s_test , y_s_test)
        RMSEC,ycal_predict=sbc.predict(bias_y,coefs_B_m,self.X_s_cal,y_s_cal)
        
        return RMSEC,RMSEP,yte_predict,comp_best_m, ycal_predict
        
    def msc(self):
        y_m_cal = deal_4_1(self.y_m_cal, self.i)
        y_s_test = deal_4_1(self.y_s_test, self.i)
        y_s_cal = deal_4_1(self.y_s_cal, self.i)
        msc=MSC(self.X_m_cal,y_m_cal,self.X_s_test,y_s_test,self.max_folds,self.max_components)
        
        coefs_B,X_trans_test,comp_best=msc.transform_train(self.X_s_test)
        RMSEP,y_predict=msc.predict(coefs_B , self.X_s_test , y_s_test,X_trans_test)
        
        coefs_B,X_trans_cal,comp_best=msc.transform_train(self.X_s_cal)
        RMSEC,ycal_predict=msc.predict(coefs_B , self.X_s_cal , y_s_cal,X_trans_cal)
        
        return RMSEC,RMSEP,y_predict,comp_best,ycal_predict
        
    def pds(self,init_width,max_width ,n_folds_cal , n_max_components_cal):
        y_m_cal = deal_4_1(self.y_m_cal, self.i)
        y_m_std = deal_4_1(self.y_m_std, self.i)
        y_s_std = deal_4_1(self.y_s_std, self.i)
        y_s_test = deal_4_1(self.y_s_test, self.i)
        y_s_cal = deal_4_1(self.y_s_cal, self.i)
        pds=PDS(self.X_m_cal,self.X_s_cal,y_m_cal,self.X_m_std,self.X_s_std,self.X_s_test,y_s_test,init_width,max_width)
#         print n_folds_cal , self.max_components , "n_fold_cal_max_components"
        Trans_matrix_list=pds.transform3(n_folds_cal,n_max_components_cal)
        best_width,best_trans_matrix,coefs_B_cal,err_list, RMSECV_list,comp_best=pds.cv_window(Trans_matrix_list,self.max_folds , self.max_components)
        print best_width , "best_width"
        y_predict,RMSEP=pds.predict(best_trans_matrix, coefs_B_cal,self.X_s_test,y_s_test)
        ycal_predict,RMSEC=pds.predict(best_trans_matrix, coefs_B_cal,self.X_s_cal,y_s_cal)
        print RMSEC , RMSEP , "RMSECP"
#         best_width, best_trans_matrix, coefs_B_cal, err_list, RMSECV_cal, comp_best_cal = pds.cv_window(Trans_matrix_list, n_folds_cal, max_components_cal)
        
        return RMSEC,best_width,RMSEP,y_predict,RMSECV_list,comp_best, ycal_predict
        
    def cca(self):
        y_m_cal = deal_4_1(self.y_m_cal, self.i)
        y_m_std = deal_4_1(self.y_m_std, self.i)
        y_s_std = deal_4_1(self.y_s_std, self.i)
        y_s_test = deal_4_1(self.y_s_test, self.i)
        y_s_cal = deal_4_1(self.y_s_cal, self.i)
        cca = CCACT(self.X_m_cal,self.X_s_cal,y_m_cal,self.X_m_std,self.X_s_std,y_s_std,self.X_m_test,self.X_s_test,y_s_test,self.max_folds,self.max_components)    
        coefficient, comp_best, RMSEC_ = cca.fit()   
        RMSEP,y_predict=cca.predict_train(self.X_s_test,y_s_test,coefficient)
        RMSEC,ycal_predict=cca.predict_train(self.X_s_cal,y_s_cal,coefficient)
        
        return RMSEC, RMSEP,y_predict,comp_best, ycal_predict
        
        
    def append(self , str ,rmsec , rmsep , y_cal_pre , y_pre , comp_best):
        self.method_list.append(str)
        self.RMSEC_list.append(rmsec)
        self.RMSEP_list.append(rmsep)
        self.ycal_predict_list.append(y_cal_pre)
        self.y_predict_list.append(y_pre)
        self.comp_best_list.append(comp_best)
    def affine_pls2(self , j ):
        pls_demo = Partial_LS(self.X_m_cal, self.y_m_cal, folds=10 , max_comp=15)
        W_m_cal, T_m_cal, P_m_cal, comp_best, coefs_cal, RMSECV = pls_demo.pls2_fit()
        
        RMSEC, RMSEP , ytrain_pre , ytest_pre = pls_demo.pls_pre_train_test(self.X_m_test, self.y_m_test, coefs_cal)
        
        print RMSEP[self.i], "RMSEP.SELF.I"
        
        
        y_ = deal_4_2(self.y_m_cal, self.i, j)
        coef = deal_4_2(coefs_cal, self.i, j)
        y_s_test_ = deal_4_2(self.y_s_test, self.i, j)
        y_s_cal_ = deal_4_2(self.y_s_cal, self.i, j)
        T_m_std, T_s_std, y_m_pre, y_s_pre, T_s_test, ys_test_pre = Pre_deal(self.X_m_cal, self.X_s_cal, W_m_cal, P_m_cal, coef, self.X_m_cal, y_, self.X_s_test)
        y0_m = y_m_pre[:, 0:1]
        y1_m = y_m_pre[:, 1:2]
        y0_s = y_s_pre[:, 0:1]
        y1_s = y_s_pre[:, 1:2]
        y0_s_test = ys_test_pre[:, 0:1]
        y1_s_test = ys_test_pre[:, 1:2]
        
        demo = Affine_trans1(y0_m, y0_s, y1_m, y1_s, comp_best)
        bia, sin_x, cos_x, x, b_s , k_m , b_m  = demo.AT_train()   
    #     print bia, sin_x, cos_x, b_s
        RMSEP_y0, RMSEP_y1 , y0_pre = demo.AT_pre(y0_s_test, y1_s_test, bia, cos_x, sin_x, y_s_test_, b_s)
        RMSEC_y0, RMSEC_y1 , y_cal_pre = demo.AT_pre(y0_s, y1_s, bia, cos_x, sin_x, y_s_cal_, b_s)       
        RMSEP_y0_,RMSEP_y1_,y0_pre_ = demo.AT_pre_no_trans(y0_s_test, y1_s_test, y_s_test_)
#         print 'RMSEP_y0/RMSEP_y1:', RMSEP_y0, RMSEP_y1
        
        
#         print ytrain_pre[:,[self.i]] , ytest_pre[:,[self.i]]
        return comp_best,RMSEP_y0,y0_pre,RMSEC_y0,y_cal_pre , y0_s_test , y1_s_test, RMSEP_y0_ , RMSEC[self.i] , RMSEP[self.i] , ytrain_pre[:,[self.i]] , ytest_pre[:,[self.i]]
        

        
    
def exe_all_corn(i , k):
    fileName = './result/result_corn_recalibration_PLS.xlsx'
    excel_result = Excel_format(fileName)   
     
    F = True
     
    master_name_o ='corn-'+bytes(i)
    print master_name_o
     
     
    n_folds_cal = 5
    max_width = 16
    init_width = 3         
    max_folds = 10
    max_components_cal = 3
    max_components = 15
    RMSEC_list = []
    RMSEP_list = []
    ycal_predict_list = []
    y_predict_list = []
    comp_best_list = []
    oplsct_comp_best_pls_list = []
    std_list = []
    h_list = []
    p_list = []
    all_list = []  
    method_list = []
    change_rate_list = []
    p_value_list = [] 
    
    std_num = 8
    
    ########## ҩƬ 10 11 12
    
    dataImport = Dataset_Import(type = 14, std_num = 8 , bool = k)
    X_master , X_slave , y = dataImport.dataset_return()
    num = int(X_master.shape[0]*0.8)
    
    X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std = Dataset_KS_split_std(X_master , X_slave , y , num , std_num)
    
    result_dict = {}
    master_name = str(master_name_o) + '_' + str(np.shape(y_m_std)[0]) 
    
    
    std_list.append(np.shape(y_m_std[:,[i]])[0])
    ycal_predict_list.append(y_s_cal[:,[i]])        
    y_predict_list.append(y_s_test[:,[i]])
    
    
#         ��һ�ַ���Ӧ�������±궨
    demo=DEMO(X_m_cal, y_m_cal, X_s_cal, y_s_cal,
               X_s_test, y_s_test, X_m_test, 
              y_m_test , X_m_std , y_m_std , 
              X_s_std , y_s_std,max_folds,
              max_components ,method_list , RMSEC_list,RMSEP_list,ycal_predict_list,y_predict_list
              ,comp_best_list , i)
            
    cal_num = y.shape[1]
    for j in range(cal_num):
        if(i == 0 and j ==1) or (i ==1 and j ==3) or (i == 2 and j == 3) or ( i == 3 and j == 2):
            affine_comp_best,affine_RMSEP,affine_y_predict,affine_RMSEC,affine_ycal_predict , y0_s_test , y1_s_test , RMSEP_y0_ ,  master_RMSEC, master_RMSEP ,master_ytrain_pre , master_ytest_pre = demo.affine_pls2(j)
            demo.append("affine-transformation2" , affine_RMSEC[0] , affine_RMSEP[0] ,affine_ycal_predict[:,[0]] ,affine_y_predict , affine_comp_best)    
            demo.append("PLS_master" , master_RMSEC , master_RMSEP , master_ytrain_pre , master_ytest_pre , affine_comp_best)
            print affine_RMSEP
            
            
            
            
    rc_comp_best,rc_RMSECV,rc_RMSEP,rc_y_predict,rc_RMSEC,rc_ycal_predict=demo.Recalibration()
               
    demo.append("Recalibration" , rc_RMSEC[0] , rc_RMSEP[0] , rc_ycal_predict, rc_y_predict,rc_comp_best)

    method_exclude = 1
       
#     ����������Լ�wilcoxon
    count = len(method_list) - method_exclude
    minIndex ,RMSEP_affines = 0 , RMSEP_list[0]
    for i in range(count):
        p , h = evaluation(RMSEP_affines , RMSEP_list[i+method_exclude] , y_predict_list[0] , y_predict_list[minIndex + 1] , y_predict_list[i+method_exclude+1])
#         print RMSEP_list[i+method_exclude] , "gg"
        change_rate_list.append(h)
        p_value_list.append(p)
          
#     print p_value_list , "p_value_list"
    result_dict={"change_rate_list":change_rate_list , "p_value_list" : p_value_list ,
            'method_list':method_list, 'std_list':std_list,
                 'RMSEC_list':RMSEC_list, 'RMSEP_list':RMSEP_list, 
                 'y_predict_list':y_predict_list, 'comp_best_list':comp_best_list,
                 'ycal_predict_list':ycal_predict_list
                 }
    print p_value_list
#     print RMSEP_list
#     print change_rate_list
    print method_list
#     print RMSEP_list , change_rate_list
    excel_result.write_excel(master_name_o, result_dict  , method_exclude)
def exe_all_yaopian(i , k):
    fileName = './result/result_corn_recalibration_PLS.xlsx'
    excel_result = Excel_format(fileName)   
     
    F = True
     
    master_name_o ='yaopian-'+bytes(i)
    print master_name_o
     
     
    n_folds_cal = 5
    max_width = 16
    init_width = 3         
    max_folds = 10
    max_components_cal = 3
    max_components = 15
    RMSEC_list = []
    RMSEP_list = []
    ycal_predict_list = []
    y_predict_list = []
    comp_best_list = []
    oplsct_comp_best_pls_list = []
    std_list = []
    h_list = []
    p_list = []
    all_list = []  
    method_list = []
    change_rate_list = []
    p_value_list = [] 
    
    std_num = 8
    
    ########## ҩƬ 10 11 12
    
    dataImport = Dataset_Import(type = 13, std_num = 8 , bool = k)
    X_master , X_slave , y = dataImport.dataset_return()
    num = int(X_master.shape[0]*0.8)
    
    X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std = Dataset_KS_split_std(X_master , X_slave , y , num , std_num)
    
    result_dict = {}
    master_name = str(master_name_o) + '_' + str(np.shape(y_m_std)[0]) 
    
    
    std_list.append(np.shape(y_m_std[:,[i]])[0])
    ycal_predict_list.append(y_s_cal[:,[i]])        
    y_predict_list.append(y_s_test[:,[i]])
    
    
#         ��һ�ַ���Ӧ�������±궨
    demo=DEMO(X_m_cal, y_m_cal, X_s_cal, y_s_cal,
               X_s_test, y_s_test, X_m_test, 
              y_m_test , X_m_std , y_m_std , 
              X_s_std , y_s_std,max_folds,
              max_components ,method_list , RMSEC_list,RMSEP_list,ycal_predict_list,y_predict_list
              ,comp_best_list , i)
            
    cal_num = y.shape[1]
    for j in range(cal_num):
        if (i ==1 and j ==2) or (i == 2 and j == 1) :
            affine_comp_best,affine_RMSEP,affine_y_predict,affine_RMSEC,affine_ycal_predict , y0_s_test , y1_s_test , RMSEP_y0_ ,  master_RMSEC, master_RMSEP ,master_ytrain_pre , master_ytest_pre = demo.affine_pls2(j)
            demo.append("affine-transformation2" , affine_RMSEC[0] , affine_RMSEP[0] ,affine_ycal_predict[:,[0]] ,affine_y_predict , affine_comp_best)    
            demo.append("PLS_master" , master_RMSEC , master_RMSEP , master_ytrain_pre , master_ytest_pre , affine_comp_best)
            print affine_RMSEP
            
            
            
            
    rc_comp_best,rc_RMSECV,rc_RMSEP,rc_y_predict,rc_RMSEC,rc_ycal_predict=demo.Recalibration()
               
    demo.append("Recalibration" , rc_RMSEC[0] , rc_RMSEP[0] , rc_ycal_predict, rc_y_predict,rc_comp_best)

    method_exclude = 1
       
#     ����������Լ�wilcoxon
    count = len(method_list) - method_exclude
    minIndex ,RMSEP_affines = 0 , RMSEP_list[0]
    for i in range(count):
        p , h = evaluation(RMSEP_affines , RMSEP_list[i+method_exclude] , y_predict_list[0] , y_predict_list[minIndex + 1] , y_predict_list[i+method_exclude+1])
#         print RMSEP_list[i+method_exclude] , "gg"
        change_rate_list.append(h)
        p_value_list.append(p)
          
#     print p_value_list , "p_value_list"
    result_dict={"change_rate_list":change_rate_list , "p_value_list" : p_value_list ,
            'method_list':method_list, 'std_list':std_list,
                 'RMSEC_list':RMSEC_list, 'RMSEP_list':RMSEP_list, 
                 'y_predict_list':y_predict_list, 'comp_best_list':comp_best_list,
                 'ycal_predict_list':ycal_predict_list
                 }
    print p_value_list
#     print RMSEP_list
#     print change_rate_list
    print method_list
#     print RMSEP_list , change_rate_list
    excel_result.write_excel(master_name_o, result_dict  , method_exclude)

if __name__ == '__main__':
    exe_all_corn(0,0)
    exe_all_corn(1,0)
    exe_all_corn(2,0)
    exe_all_corn(3,0)
    exe_all_yaopian(1, 1)
    exe_all_yaopian(2, 1)
    ####### ��3���ɷֲ��ã���2���ɷ��á���4���ɷ��� �� ��һ���ɷ���
         
          