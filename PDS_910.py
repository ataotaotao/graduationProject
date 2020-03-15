# -*-coding:utf-8-*-
'''
Created on 2016��9��1��

@author: Administrator
'''

from __future__ import division
import numpy as np
from pls_demo import PlsDemo
from sympy.integrals.heurisch import components
# import drawings 


class PDS():
    def __init__(self, x_src_cal, x_tar_cal, y_src_cal, x_src_std, x_tar_std, x_tar_test, y_tar_test, init_width, max_width):
        self.x_src_cal = x_src_cal
        self.x_tar_cal = x_tar_cal
        self.y_src_cal = y_src_cal
        self.x_src_std = x_src_std
        self.x_tar_std = x_tar_std
#         self.x_src_test = x_src_test
        self.x_tar_test = x_tar_test
        self.y_tar_test = y_tar_test
        self.init_width = init_width
        self.max_width = max_width
        
    def transform2(self, n_folds, n_components):
        x_src_std_mean = np.mean(self.x_src_std, axis=0)
        x_tar_std_mean = np.mean(self.x_tar_std, axis=0)
        x_src_std_center = np.subtract(self.x_src_std, x_src_std_mean)
        x_tar_std_center = np.subtract(self.x_tar_std, x_tar_std_mean)
        n, m = np.shape(self.x_src_std)
#         print 'n,m:', n, m
        Trans_matrix_list = []
        for i in range(self.init_width, self.max_width, 2):
            Trans_matrix = np.zeros((m, m))
            j = (i - 1) // 2  # hemi_width
##########�����������ݣ��ԳƷ�ʽ��ģ
            for k in range(j, m - j - 1):
#                 print 'k:', k
#                 pls=PlsDemo(self.x_tar_std[:,k-j:k+j+1],self.x_src_std[:,[k]],n_folds,max_components)
#                 W,T,coefs_B,RMSECV,min_RMSECV,comp_best=pls.pls_fit()
                demo1 = PlsDemo(x_tar_std_center[:, k - j:k + j + 2], x_src_std_center[:, [k]], n_folds, n_components)
                W3, T3, P3, coefs_B3, RMSECV3, min_RMSECV3, comp_best3 = demo1.pls_fit()
#                print np.shape(coefs_B)
  
#                print np.shape(Trans_matrix[k:k+i,k])
                Trans_matrix[k - j:k + j + 2, k] = coefs_B3.ravel()
            n_components = n_components + 2
            Trans_matrix_list.append(Trans_matrix)

        return Trans_matrix_list
    def transform3(self, n_folds, n_components):
        x_src_std_mean = np.mean(self.x_src_std, axis=0)
        x_tar_std_mean = np.mean(self.x_tar_std, axis=0)
#         ������ֵ����
        x_src_std_center = np.subtract(self.x_src_std, x_src_std_mean)
        x_tar_std_center = np.subtract(self.x_tar_std, x_tar_std_mean)
        n, m = np.shape(self.x_src_std)
        Trans_matrix_list = []
        for i in range(self.init_width, self.max_width, 2):
            Trans_matrix = np.zeros((m, m))
            j = (i - 1) // 2  # hemi_width
            l1 = 0
            l2 = 2 * j + 1
            l3 = m - 2 * j - 1
            l4 = m
            
            for k in range(0, m):
                 
                if(k < j):
                    demo1 = PlsDemo(x_tar_std_center[:, l1:l2], x_src_std_center[:, [k]], n_folds, n_components)
                    W1, T1, P1, coefs_B1, RMSECV1, min_RMSECV1, comp_best1 = demo1.pls_fit()
                    Trans_matrix[l1:l2, k] = coefs_B1.ravel()
                    l1 = l1
                    l2 = l2 + 1
                    
                if(k >= m - j):
                    
                    demo2 = PlsDemo(x_tar_std_center[:, l3:l4], x_src_std_center[:, [k]], n_folds, n_components)
                    W2, T2, P2, coefs_B2, RMSECV2, min_RMSECV2, comp_best2 = demo2.pls_fit()
                    Trans_matrix[l3:l4, k] = coefs_B2.ravel()
                    l3 = l3 - 1
                    l4 = l4
                    
                if(j <= k < m - j):
                    demo1 = PlsDemo(x_tar_std_center[:, k - j:k + j + 1], x_src_std_center[:, [k]], n_folds, n_components)
                    W3, T3, P3, coefs_B3, RMSECV3, min_RMSECV3, comp_best3 = demo1.pls_fit()
                    Trans_matrix[k - j:k + j + 1, k] = coefs_B3.ravel()
            n_components = n_components + 2       
            Trans_matrix_list.append(Trans_matrix)

        return Trans_matrix_list

    def cv_window(self, Trans_matrix_list, n_folds_cal, max_components_cal):
        pls = PlsDemo(self.x_src_cal, self.y_src_cal, n_folds_cal, max_components_cal)
        W_cal, T_cal, P_cal, coefs_B_cal, RMSECV_cal, min_RMSECV_cal, comp_best_cal = pls.pls_fit()
        
        xcal_src_mean = np.mean(self.x_src_cal, axis=0)
        ycal_src_mean = np.mean(self.y_src_cal, axis=0)
        
        x_src_std_mean = np.mean(self.x_src_std, axis=0)
        x_tar_std_mean = np.mean(self.x_tar_std, axis=0)
        
        x_tar_cal_center = np.subtract(self.x_tar_cal, x_tar_std_mean)

        l = len(Trans_matrix_list)
#         print l,"llllllllll"
        err_list = []
        for i in range(l):
            trans_matrix = Trans_matrix_list[i]
            x_tar_cal_trans = np.dot(x_tar_cal_center, trans_matrix) + x_src_std_mean
            x_tar_cal_trans_center = np.subtract(x_tar_cal_trans, xcal_src_mean)
            y_tar_cal_trans_pre = np.dot(x_tar_cal_trans_center, coefs_B_cal) + ycal_src_mean
            
            err_cal = np.sqrt(np.sum(np.square(np.subtract(self.y_src_cal, y_tar_cal_trans_pre)), axis=0) / self.x_tar_cal.shape[0])
            err_list.append(err_cal)

        index_value = err_list.index(min(err_list))
        best_width = self.init_width + 2 * index_value
        best_trans_matrix = Trans_matrix_list[index_value]
#         print min(err_list)
#         print best_width
#         print best_width , "best_width"
        return best_width, best_trans_matrix, coefs_B_cal, err_list, RMSECV_cal, comp_best_cal
    
    def predict(self, best_trans_matrix, coefs_B_cal, Xs_test, ys_test):
       
        x_src_cal_mean = np.mean(self.x_src_cal, axis=0)
        y_src_cal_mean = np.mean(self.y_src_cal, axis=0)
        
        x_src_std_mean = np.mean(self.x_src_std, axis=0)
        x_tar_std_mean = np.mean(self.x_tar_std, axis=0)
        
        x_tar_test_center = np.subtract(Xs_test, x_tar_std_mean)
        x_tar_test_trans = np.dot(x_tar_test_center, best_trans_matrix) + x_src_std_mean
        x_tar_trans_center = np.subtract(x_tar_test_trans, x_src_cal_mean)
        y_tar_pre = np.dot(x_tar_trans_center, coefs_B_cal) + y_src_cal_mean
        
        RMSEP = np.sqrt(np.sum(np.square(np.subtract(ys_test, y_tar_pre)), axis=0) / Xs_test.shape[0])
#        drawings.draws_pre(self.y_tar,y_tar_pre)
          
        return y_tar_pre, RMSEP
        
        
  
if __name__ == '__main__':
    import numpy as np
    from scipy import linalg
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    

    
#     fname=loadmat('Pharmaceutical tablet')
#     D=fname
#     print D.keys()
#      
#     x_src_cal=D['calibrate_1']['data'][0][0]
#     print np.shape(x_src_cal)
#       
#     x_tar_cal=D['calibrate_2']['data'][0][0]
#     y_src_cal=D['calibrate_Y']['data'][0][0][:,1:2]
#     y_tar_cal=y_src_cal
#       
#     x_src_std=D['validate_1']['data'][0][0]
#     print np.shape(x_src_std)
#     x_tar_std=D['validate_2']['data'][0][0]
#   
#     x_src_test=D['test_1']['data'][0][0]
#     print np.shape(x_src_test)
#     x_tar_test=D['test_2']['data'][0][0]
#     y_tar_test=D['test_Y']['data'][0][0][:,1:2]
  
#     fname = loadmat('NIRcorn.mat')
# #     print fname.keys()
# # #     X_master = fname['cornspect']
#     X_master = fname['m5spec']['data'][0][0]
#     y = fname['cornprop'][:, 3:4]
#     X_slave = fname['mp6spec']['data'][0][0]  

    fname = loadmat('wheat_A_cal.mat')  # ���ɷ֣�5��window:3
    print fname.keys()
    X_master = fname['CalSetA2']
    y = fname['protein']
    X_slave = fname['CalSetA3']


#     fname = loadmat('wheat_B_cal.mat')   ���ɷ֣�10��window:2
#     print fname.keys()
#     X_master = fname['CalSetB1']
#     X_slave = fname['CalSetB3']
#     y = fname['protein']
    print np.shape(X_master), np.shape(X_slave)
    N_list = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#     0.1, 0.2, 0.3, 0.4,
    
    for i in N_list:
        print '\n\n            ��ǰ�ٷֱ�Ϊ%i,', (i)
        
        '''        ���ݼ��Ļ���                '''  
        x_src_cal, x_src_test, y_src_cal, y_src_test = train_test_split(X_master, y, test_size=0.2, random_state=0)
        x_tar_cal, x_tar_test, y_tar_cal, y_tar_test = train_test_split(X_slave, y, test_size=0.2, random_state=0)
        x_src_train, x_src_std, y_src_train, y_src_std = train_test_split(x_src_cal, y_src_cal, test_size=i, random_state=0)
        x_tar_train, x_tar_std, y_tar_train, y_tar_std = train_test_split(x_tar_cal, y_tar_cal, test_size=i, random_state=0)
    
        n_folds = 10
        max_components = 3  # �ֲ�ģ��
        max_width = 16
        init_width = 3
        n_folds_cal = 10
        max_components_cal = 5
        
        pds = PDS(x_src_cal, x_tar_cal, y_src_cal, x_src_std, x_tar_std, x_src_test, x_tar_test, y_tar_test, init_width, max_width)
        
        Trans_matrix_list = pds.transform3(n_folds, max_components)
      
        best_width, best_trans_matrix, coefs_B_cal, err_list, RMSECV_cal, comp_best_cal = pds.cv_window(Trans_matrix_list, n_folds_cal, max_components_cal)
   
        y_tar_pre, RMSEP = pds.predict(best_trans_matrix, coefs_B_cal)
    
        print best_width
        print RMSEP
                
            
            
            
        
        
