#-*-coding:utf-8-*-
'''
Created on 2018��8��14��

@author: DELL
'''
from __future__ import division
from sklearn import cross_validation 
from scipy.stats import f
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSRegression

from pls_demo import PlsDemo

'''
Calibration model transfer for near-infrared spectra based on canonical correlation analysis
'''
class baseCV():   
    def __init__(self, x,y):
        self.x = x
        self.y= y
        self.n = self.x.shape[0]

    def CV(self, x, y, n_folds):
        kf = cross_validation.KFold(self.n, n_folds)

        x_tr = []
        y_tr=[]
        x_te = []
        y_te = []
        
        for train_index, test_index in kf:

            xtr, xte = self.x[train_index], self.x[test_index]
            ytr, yte = self.y[train_index], self.y[test_index]
 
            x_tr.append(xtr)
            x_te.append(xte)
            y_tr.append(ytr)
            y_te.append(yte)
 
        return x_tr, x_te, y_tr, y_te #��Ϊ�б�    
    
class PLSCV(baseCV):        
    def __init__(self, x, y):
        baseCV.__init__(self, x, y)

    def cv_predict(self, n_folds, max_components):
                 
        x_tr, x_te, y_tr, y_te = baseCV.CV(self, self.x, self.y, n_folds)

        y_predict_all=np.ones((1,max_components))
#         pls = _NIPALS(max_components)
        for i in range(n_folds):
            
            y_predict = np.zeros((x_te[i].shape[0],max_components))
            xtrainmean = np.mean(x_tr[i], axis=0)
            ytrainmean = np.mean(y_tr[i], axis=0)

            xte_center = np.subtract(x_te[i], xtrainmean)                              
            yte_center = np.subtract(y_te[i],ytrainmean)

#             W,T,P,C,U,Q,lists_coefs_B=pls.fit(x_tr[i],y_tr[i],max_components)
#            print lists_coefs_B
            for j in range(1, max_components, 1):

                pls2 = PLSRegression(j)
                pls2.fit(x_tr[i],y_tr[i])
#                 print pls2.coef_
        
                y_pre_center = np.dot(xte_center,pls2.coef_)
#                 y_pre_center = np.dot(xte_center,lists_coefs_B[j])
                Y_pre = y_pre_center + ytrainmean                                
                y_predict[:,j]=Y_pre.ravel() 
            
            y_predict_all=np.vstack((y_predict_all,y_predict))
            
        y_predict_all=y_predict_all[1:]
 
        return   y_predict_all,self.y
    
    def cv_mse(self,Y_predict_all,y):
       
        press=np.square(np.subtract(Y_predict_all,y))
        PRESS_all=np.sum(press,axis=0)
        RMSECV_array=np.sqrt(PRESS_all/self.n)
        min_RMSECV=min(RMSECV_array)
        comp_array=RMSECV_array.argsort()
        comp_best=comp_array[0]+1

        return  RMSECV_array,min_RMSECV,comp_best
    
    def cv_mse_f(self,Y_predict_all,y,alpha):
           
        press=np.square(np.subtract(Y_predict_all,y))
        PRESS_all=np.sum(press,axis=0) 
        RMSECV_array=np.sqrt(PRESS_all/self.n)
        comp_array=RMSECV_array.argsort()
        comp_best=comp_array[0]+1
        F_value=f.isf(alpha,self.n-1,self.n-1)
        k = 0
        while RMSECV_array[k] > (RMSECV_array[comp_best-1] * np.sqrt(F_value)):
            k = k+1
#         if k == 0:
#             k = 1
        min_comp_best = k + 1
            
        min_RMSECV=RMSECV_array[min_comp_best-1]
     
        return  RMSECV_array,min_RMSECV,min_comp_best
    
class CCACT():
    def __init__(self,x_m_cal,x_s_cal,y_cal,x_m_std,x_s_std,y_std,x_m_test,x_s_test,y_test,max_folds,max_components):
        
        self.x_m_cal=x_m_cal
        self.x_s_cal=x_s_cal
        self.y_cal=y_cal
        self.x_m_std=x_m_std
        self.x_s_std=x_s_std
        self.y_std=y_std
        self.x_m_test=x_m_test
        self.x_s_test=x_s_test
        self.y_test=y_test
        self.max_folds=max_folds
        self.max_components=max_components
     
    def fit(self):             
        
        # PlsDemo
        PD_m=PlsDemo(self.x_m_cal,self.y_cal,self.max_folds,self.max_components)       
        W_m,T_m,P_m,coefs_B_m,RMSECV_m,min_RMSECV_m,comp_best=PD_m.pls_fit() 
#         print "comp_best =", comp_best
        
#         cca = CCA(comp_best)
#         cca.fit(self.x_m_std, self.y_std)
#         X_score, Y_score = cca.transform(self.x_m_std, self.y_std)     
#         W_m = cca.x_weights_
#         P_m = cca.x_loadings_
#         W = np.dot(cca.x_weights_, linalg.pinv2(np.dot(cca.x_loadings_.T, cca.x_weights_)))
#         print "cca =", cca
#         print cca.x_rotations_
#         print W 
#         print cca.x_scores_
#         print X_score
#         print np.dot(np.subtract(self.x_m_std, self.x_m_std.mean(0)), cca.x_rotations_)
#         print np.dot(np.subtract(self.x_m_std, self.x_m_std.mean(0)), W)
        
        cca_m = CCA(comp_best)
        cca_m.fit(self.x_m_std, self.y_std)
        X_score, Y_score = cca_m.transform(self.x_m_std, self.y_std)
        W_m = cca_m.x_rotations_
        x_m_std_mean = np.mean(self.x_m_std, axis=0)
        x_m_std_center = np.subtract(self.x_m_std, x_m_std_mean)
        L_m = np.dot(x_m_std_center, W_m)
        print self.x_m_std.shape , W_m.shape , "shape"
        
        cca_s = CCA(comp_best)
        cca_s.fit(self.x_s_std, self.y_std)
        X_score, Y_score = cca_s.transform(self.x_s_std, self.y_std)
        W_s = cca_s.x_rotations_
        x_s_std_mean = np.mean(self.x_s_std, axis=0)
        x_s_std_center = np.subtract(self.x_s_std, x_s_std_mean)
        L_s = np.dot(x_s_std_center, W_s)
        print self.x_s_std.shape , W_s.shape , "shape"
#         print "L.shape =", np.shape(L_m),np.shape(L_s)
        
        F_1 = np.linalg.lstsq(L_s, L_m)[0]
        F_2 = np.linalg.lstsq(L_m, self.x_m_std)[0]

        coefficient = np.dot(np.dot(np.dot(W_s, F_1), F_2), coefs_B_m)
        
        #RMSEC
#         xs_std_center=np.subtract(self.x_s_std, self.x_s_std.mean(axis=0))
# #         xs_std_center=np.subtract(self.x_s_std, self.x_m_cal.mean(axis=0))
#         y_predict=np.dot(xs_std_center, coefficient)+self.y_cal.mean(axis=0)
#         RMSEC=np.sqrt(np.sum(np.square(np.subtract(y_predict,self.y_std)),axis=0)/self.y_std.shape[0])
#         print "RMSEC =", RMSEC
        
        xs_cal_center=np.subtract(self.x_s_cal, self.x_s_cal.mean(axis=0))
#         xs_cal_center=np.subtract(self.x_s_cal, self.x_m_cal.mean(axis=0))
        y_predict=np.dot(xs_cal_center, coefficient)+self.y_cal.mean(axis=0)
        RMSEC=np.sqrt(np.sum(np.square(np.subtract(y_predict,self.y_cal)),axis=0)/self.y_cal.shape[0])
        
#         # PLSRegression
#         pls_cv=PLSCV(self.x_m_cal,self.y_cal)                   
#         y_predict_all,y_measure=pls_cv.cv_predict(self.max_folds,self.max_components)
#         
#         if F:
#             RMSECV,min_RMSECV,comp_best=pls_cv.cv_mse_f(y_predict_all, y_measure, alpha=0.05)
#         else:
#             RMSECV,min_RMSECV,comp_best=pls_cv.cv_mse(y_predict_all, y_measure)
#         print "comp_best =", comp_best
#         
#         plsr = PLSRegression(j)
#         plsr.fit(self.x_m_cal,self.y_cal)
#         coefs_B = plsr.coef_
#         coefficient = np.dot(np.dot(np.dot(W_s, F_1), F_2), coefs_B)  
        
#         xs_test_center=np.subtract(self.x_s_test, self.x_s_test.mean(axis=0))
#         y_predict=np.dot(xs_test_center, np.dot(np.dot(np.dot(W_s, F_1), F_2), coefs_B_m))+self.y_cal.mean(axis=0)      
#         RMSEP=np.sqrt(np.sum(np.square(np.subtract(self.y_test,y_predict)),axis=0)/self.y_test.shape[0])
#         print "PlsDemo =", RMSEP
        
        return coefficient, comp_best, RMSEC

    def fit_cal(self,F):             
        
        # PlsDemo
        PD_m=PlsDemo(self.x_m_cal,self.y_cal,self.max_folds,self.max_components)       
        W_m,T_m,P_m,coefs_B_m,RMSECV_m,min_RMSECV_m,comp_best=PD_m.pls_fit(F) 
#         print "comp_best =", comp_best
        
        cca_m = CCA(comp_best)
        cca_m.fit(self.x_m_cal, self.y_cal)
        X_score, Y_score = cca_m.transform(self.x_m_cal, self.y_cal)
        W_m = cca_m.x_rotations_      
        x_m_cal_mean = np.mean(self.x_m_cal, axis=0)
        x_m_cal_center = np.subtract(self.x_m_cal, x_m_cal_mean)
        L_m = np.dot(x_m_cal_center, W_m)
        
        cca_s = CCA(comp_best)
        cca_s.fit(self.x_s_std, self.y_std)
        X_score, Y_score = cca_s.transform(self.x_s_std, self.y_std)
        W_s = cca_s.x_rotations_
        x_s_std_mean = np.mean(self.x_s_std, axis=0)
        x_s_std_center = np.subtract(self.x_s_std, x_s_std_mean)
        L_s = np.dot(x_s_std_center, W_s)
        
#         print "L.shape =", np.shape(L_m),np.shape(L_s)
        
        F_1 = np.linalg.lstsq(L_s, L_m)[0]
        F_2 = np.linalg.lstsq(L_m, self.x_m_std)[0]

        coefficient = np.dot(np.dot(np.dot(W_s, F_1), F_2), coefs_B_m)
        
        #RMSEC
#         xs_std_center=np.subtract(self.x_s_std, self.x_s_std.mean(axis=0))
# #         xs_std_center=np.subtract(self.x_s_std, self.x_m_cal.mean(axis=0))
#         y_predict=np.dot(xs_std_center, coefficient)+self.y_cal.mean(axis=0)
#         RMSEC=np.sqrt(np.sum(np.square(np.subtract(y_predict,self.y_std)),axis=0)/self.y_std.shape[0])
#         print "RMSEC =", RMSEC
        
        xs_cal_center=np.subtract(self.x_s_cal, self.x_s_cal.mean(axis=0))
#         xs_cal_center=np.subtract(self.x_s_cal, self.x_m_cal.mean(axis=0))
        y_predict=np.dot(xs_cal_center, coefficient)+self.y_cal.mean(axis=0)
        RMSEC=np.sqrt(np.sum(np.square(np.subtract(y_predict,self.y_cal)),axis=0)/self.y_cal.shape[0])

        return coefficient, comp_best, RMSEC
    
    def fit_m(self,F):             
        
        # PlsDemo
        PD_m=PlsDemo(self.x_m_cal,self.y_cal,self.max_folds,self.max_components)       
        W_m,T_m,P_m,coefs_B_m,RMSECV_m,min_RMSECV_m,comp_best=PD_m.pls_fit(F) 
#         print "comp_best =", comp_best
        
        cca_m = CCA(comp_best)
        cca_m.fit(self.x_m_cal, self.y_cal)
        X_score, Y_score = cca_m.transform(self.x_m_cal, self.y_cal)
        P_m = cca_m.x_loadings_
        W_m = cca_m.x_rotations_   
        x_m_cal_mean = np.mean(self.x_m_cal, axis=0)
        
        x_m_std_center = np.subtract(self.x_m_std, x_m_cal_mean)
        L_m = np.dot(x_m_std_center, W_m)

        x_s_std_center = np.subtract(self.x_s_std, x_m_cal_mean)
        L_s = np.dot(x_s_std_center, W_m)
        
        print "L.shape =", np.shape(L_m),np.shape(L_s)
        
        F_1 = np.linalg.lstsq(L_s, L_m)[0]
        F_2 = W_m #np.linalg.lstsq(L_m, self.x_m_cal)[0]

        coefficient = np.dot(np.dot(np.dot(W_m, F_1), P_m.T), coefs_B_m)
        
        #RMSEC
#         xs_std_center=np.subtract(self.x_s_std, self.x_s_std.mean(axis=0))
# #         xs_std_center=np.subtract(self.x_s_std, self.x_m_cal.mean(axis=0))
#         y_predict=np.dot(xs_std_center, coefficient)+self.y_cal.mean(axis=0)
#         RMSEC=np.sqrt(np.sum(np.square(np.subtract(y_predict,self.y_std)),axis=0)/self.y_std.shape[0])
#         print "RMSEC =", RMSEC
        
        xs_cal_center=np.subtract(self.x_s_cal, self.x_s_cal.mean(axis=0))
#         xs_cal_center=np.subtract(self.x_s_cal, self.x_m_cal.mean(axis=0))
        y_predict=np.dot(xs_cal_center, coefficient)+self.y_cal.mean(axis=0)
        RMSEC=np.sqrt(np.sum(np.square(np.subtract(y_predict,self.y_cal)),axis=0)/self.y_cal.shape[0])

        return coefficient, comp_best, RMSEC
    
    def predict(self, coefficient):
        
#         xs_test_center=np.subtract(self.x_s_test, self.x_s_test.mean(axis=0))
        xs_test_center=np.subtract(self.x_s_test, self.x_s_std.mean(axis=0))
#         xs_test_center=np.subtract(self.x_s_test, self.x_m_cal.mean(axis=0))
 
        y_predict=np.dot(xs_test_center, coefficient)+self.y_cal.mean(axis=0)
               
        RMSEP=np.sqrt(np.sum(np.square(np.subtract(self.y_test,y_predict)),axis=0)/self.y_test.shape[0])
#         drawings.draws_pre(y_predict,self.y_test)
#         print "RMSEP =", RMSEP


        return  RMSEP,y_predict   
    def predict_train(self, x,y,coefficient):
        
#         xs_test_center=np.subtract(self.x_s_test, self.x_s_test.mean(axis=0))
        xs_test_center=np.subtract(x, self.x_s_std.mean(axis=0))
#         xs_test_center=np.subtract(self.x_s_test, self.x_m_cal.mean(axis=0))
 
        y_predict=np.dot(xs_test_center, coefficient)+self.y_cal.mean(axis=0)
               
        RMSE=np.sqrt(np.sum(np.square(np.subtract(y,y_predict)),axis=0)/y.shape[0])
#         drawings.draws_pre(y_predict,self.y_test)
#         print "RMSEP =", RMSEP


        return  RMSE,y_predict 

if __name__ == '__main__':
    import numpy as np
    from scipy import linalg
    from scipy.io import loadmat,savemat
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from openpyxl import load_workbook
    import sys
    import os
    import function_module as fm
    
    path = os.path.abspath(os.path.join(os.getcwd(), "./"))#��ȡ���ϼ�Ŀ¼
      
    mat_name =[
                '\NIRcorn.mat'
#                 '\KS_new\corn_m5-mp6_0.mat','\KS_new\corn_m5-mp6_1.mat','\KS_new\corn_m5-mp6_2.mat','\KS_new\corn_m5-mp6_3.mat',
#                 '\KS_new\corn_m5-mp5_0.mat','\KS_new\corn_m5-mp5_1.mat','\KS_new\corn_m5-mp5_2.mat','\KS_new\corn_m5-mp5_3.mat',
#                 '\KS_new\corn_mp5-mp6_0.mat','\KS_new\corn_mp5-mp6_1.mat','\KS_new\corn_mp5-mp6_2.mat','\KS_new\corn_mp5-mp6_3.mat',
#                 '\KS_new\corn_mp5-m5_0.mat','\KS_new\corn_mp5-m5_1.mat','\KS_new\corn_mp5-m5_2.mat','\KS_new\corn_mp5-m5_3.mat',
#                 '\KS_new\corn_mp6-m5_0.mat','\KS_new\corn_mp6-m5_1.mat','\KS_new\corn_mp6-m5_2.mat','\KS_new\corn_mp6-m5_3.mat',
#                 '\KS_new\corn_mp6-mp5_0.mat','\KS_new\corn_mp6-mp5_1.mat','\KS_new\corn_mp6-mp5_2.mat','\KS_new\corn_mp6-mp5_3.mat',
#                 '\KS_new\wheat_A1-A2.mat','\KS_new\wheat_A1-A3.mat','\KS_new\wheat_A2-A1.mat','\KS_new\wheat_A2-A3.mat','\KS_new\wheat_A3-A1.mat','\KS_new\wheat_A3-A2.mat',
#                 '\KS_new\wheat_B1-B2.mat','\KS_new\wheat_B1-B3.mat','\KS_new\wheat_B2-B1.mat','\KS_new\wheat_B2-B3.mat','\KS_new\wheat_B3-B1.mat','\KS_new\wheat_B3-B2.mat',
#                 '\KS_new\pharmceutical_tablet_1-2_0.mat','\KS_new\pharmceutical_tablet_1-2_1.mat','\KS_new\pharmceutical_tablet_1-2_2.mat',
#                 '\KS_new\pharmceutical_tablet_2-1_0.mat', '\KS_new\pharmceutical_tablet_2-1_1.mat', '\KS_new\pharmceutical_tablet_2-1_2.mat' 
            ]
    
    F = True
      
    for i in range(len(mat_name)):
        print path+mat_name[i]
        mat = loadmat(path+mat_name[i])
        master_name_o = str(mat_name[i])[8:-4]
        print "master_name_o =", master_name_o
        
        n_folds = 5
        n_components = 15
#         max_width=4
#         init_width=3  
                  
        max_folds = 10
        max_components = 15
              
              
              
              
              
              
              
              
              
              
              
              
              
        x_m = mat['mp5spec']['data'][0][0]
        x_s = mat['mp6spec']['data'][0][0]  
        y = mat['cornprop'][:,[1]]      
        #### 64 16
        num = int(round(x_m.shape[0]*0.8))
#         print "num =", num
        x_m_cal, y_cal, x_s_cal, x_s_test, x_m_test, y_test = fm.Data_KS_split(x_m,x_s,y,num)
              
        
        x_m_std = x_m_cal    
        x_s_std = x_s_cal         
        y_std = y_cal

#             print "shape :", np.shape(x_m_cal),np.shape(x_m_std),np.shape(x_m_test)
        
        master_name = str(master_name_o) + '_' + str(np.shape(y_std)[0]) 
#             print "master_name =", master_name

        
        demo = CCACT(x_m_cal, x_s_cal, y_cal, x_m_std, x_s_std, y_std, x_m_test, x_s_test, y_test, max_folds, max_components)    
        
        coefficient,comp_best, RMSEC = demo.fit()   
        RMSEP,y_predict=demo.predict(coefficient) 
        print "std:", RMSEC, RMSEP
            
#             coefficient,comp_best, RMSEC = demo.fit_cal(F=True)   
#             RMSEP,y_predict=demo.predict(coefficient)
#             print "cal:", RMSEC, RMSEP
             
#             coefficient,comp_best, RMSEC = demo.fit_m(F=True)   
#             RMSEP,y_predict=demo.predict(coefficient)             
#             print "mW:", RMSEC, RMSEP
             
            
            
            
            
            
            