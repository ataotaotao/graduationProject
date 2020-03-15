# -*-coding:utf-8-*-
'''
Created on 2016��8��31��
������Σ��������ɱ��������²�����ٿ��꣬������裬����ɷ����������ߣ���֮��Ҳ���ñ�֮��Ҳ��
����ɱ����������Σ�����ɲ���Ҳ��
@author: Administrator
'''
from __future__ import division
from sklearn import linear_model
# import pylab as pl
import numpy as np
from pls_demo import PlsDemo
# import drawings 
import function_module as fm

class MSC():
    def __init__(self, x_src_cal, y_src_cal, x_tar_test, y_tar_test, n_folds, max_components):
        self.x_src_cal = x_src_cal
        self.y_src_cal = y_src_cal
        self.x_tar_test = x_tar_test
        self.y_tar_test = y_tar_test
        self.n_folds = n_folds
        self.max_components = max_components
        
    def transform(self):
        n, m = np.shape(self.x_tar_test)

        x_src_mean = self.x_src_cal.mean(axis=0)

        
        x_mean = np.mat(x_src_mean).T
        x_tar_T = self.x_tar_test.T
        X_trans = np.ones((m, n))
        for i in range(n):
            X = x_tar_T[:, [i]]
            clf = linear_model.LinearRegression()
            clf.fit (x_mean, X)
            coef = clf.coef_
            intercept = clf.intercept_
            x_trans = np.subtract(X, intercept) / coef
            X_trans[:, i] = x_trans.ravel()
        return X_trans.T
    def transform_train(self , x):
        pls = PlsDemo(self.x_src_cal, self.y_src_cal, self.n_folds, self.max_components)
        W, T, P, coefs_B, RMSECV, min_RMSECV, comp_best = pls.pls_fit()
        
        n, m = np.shape(x)
        print x.shape
        x_src_mean = self.x_src_cal.mean(axis=0)

        
        x_mean = np.mat(x_src_mean).T
        x_tar_T = x.T
        X_trans = np.ones((m, n))
        for i in range(n):
            X = x_tar_T[:, [i]]
            clf = linear_model.LinearRegression()
            clf.fit (x_mean, X)
            coef = clf.coef_
            intercept = clf.intercept_
            x_trans = np.subtract(X, intercept) / coef
            X_trans[:, i] = x_trans.ravel()
        return coefs_B,X_trans.T,comp_best
    
    def predict(self, coef_B , x ,y , X_trans):
        x_src_mean = np.mean(self.x_src_cal, axis=0)
        y_src_mean = np.mean(self.y_src_cal, axis=0)
        X_trans_center = np.subtract(X_trans, x_src_mean)
        y_tar_predict = np.dot(X_trans_center, coef_B) + y_src_mean
        RMSEP = np.sqrt(np.sum(np.square(np.subtract(y, y_tar_predict)), axis=0) / x.shape[0])
#        print RMSEP
#        drawings.draws_pre(self.y_tar_test,y_tar_predict)
        return RMSEP, y_tar_predict
def deal_4_1(x , i):
    x_ = np.zeros((x.shape[0] , 1))
    x_[:,0] = x[:, i].ravel()
    return x_   
            
if __name__ == '__main__':
    import numpy as np
    from scipy import linalg
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from function_module import SNV

    
    fname=loadmat('Pharmaceutical tablet')
    D=fname
    print D.keys()
#             
#     x_src_cal=D['calibrate_1']['data'][0][0]
#     x_tar_cal=D['calibrate_2']['data'][0][0]
#     y_src_cal=D['calibrate_Y']['data'][0][0][:,2:3]
#     y_tar_cal=y_src_cal   
# #     x_src_std=D['validate_1']['data'][0][0]
# #     x_tar_std=D['validate_2']['data'][0][0]
#                
#     x_src_test=D['test_1']['data'][0][0]
#     x_tar_test=D['test_2']['data'][0][0]
#     y_src_test=D['test_Y']['data'][0][0][:,2:3]
#     y_tar_test=y_src_test
    
    fname = loadmat('NIRcorn.mat')
    D = fname
    print D.keys()                          
    x_src = D['cornspect']
             
    x_mp5spec = D['mp5spec']['data'][0][0]
#    x_m5spec=D['m5spec']['data'][0][0]   #x_m5spec=x
    x_mp6spec = D['mp6spec']['data'][0][0] 
    x_mp5spec = SNV(x_mp5spec)
    x_mp6spec = SNV(x_mp6spec)
    x_tar = x_mp6spec
    for i in range(4):
        y_cal = D['cornprop'][:, 0:4]
        
        num = int (x_mp5spec.shape[0] * 0.8)
        x_m_cal, y_m_cal, x_s_cal, y_s_cal, x_s_test, y_s_test, x_m_test, y_m_test , x_m_std , y_m_std , x_s_std , y_s_std = fm.Dataset_KS_split_std(x_mp5spec, x_mp6spec, y_cal, num)
#         x_m_cal, y_m_cal, x_s_cal, y_s_cal, x_s_test, y_s_test, x_m_test, y_m_test = fm.Dataset_KS_split(x_mp5spec, x_tar, y_cal, num)
        
#         x_src_cal, x_src_test, y_src_cal, y_src_test = train_test_split(x_src, y_cal, test_size=0.2, random_state=0)
#         x_tar_cal, x_tar_test, y_tar_cal, y_tar_test = train_test_split(x_tar, y_cal, test_size=0.2, random_state=0)
    #     x_src_train, x_src_std, y_src_train, y_src_std = train_test_split(x_src_cal, y_src_cal, test_size=0.5, random_state=0)
    #     x_tar_train, x_tar_std, y_tar_train, y_tar_std = train_test_split(x_tar_cal, y_tar_cal, test_size=0.5, random_state=0)
    
        n_folds = 10
        max_components = 15
            
        y_m_cal = deal_4_1(y_m_cal, i)
        y_s_test = deal_4_1(y_s_test, i)
        y_s_cal = deal_4_1(y_s_cal,i)
        msc=MSC(x_m_cal,y_m_cal,x_s_test,y_s_test,n_folds, max_components)
        
        coefs_B,X_trans_test,comp_best=msc.transform_train(x_s_test)
        RMSEP,y_predict=msc.predict(coefs_B , x_s_test , y_s_test,X_trans_test)
            
            
#         msc = MSC(x_m_cal, y_m_cal, x_s_test, y_s_test, n_folds, max_components)
#         coefs_B,X_trans_test,comp_best=msc.transform_train(x_s_test)
#         RMSEP,y_predict=msc.predict(coefs_B , x_s_test , y_s_test,X_trans_test)
        
        
#         X_trans = msc.transform()
#         RMSEP, y_tar_predict, comp_best = msc.predict(X_trans , )
        
        print RMSEP
#     print X_trans
#     print np.shape(X_trans)
#     print np.shape(x_src_cal)
#    print np.subtract(x_src_cal,X_trans)
         
