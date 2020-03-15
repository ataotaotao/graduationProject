#-*-coding:utf-8-*-
'''
Created on 2015楠烇拷0閺堬拷1閺冿拷

@author: lenovo
'''
from __future__ import division
from sklearn import cross_validation 
#import pylab as pl
import scipy.io as sio
import numpy as np
from scipy import linalg

from NIPALS import _NIPALS


 
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
#            print train_index, test_index

            xtr, xte = self.x[train_index], self.x[test_index]
#             print test_index , "test_index" , xte[0].shape
#            print xte
            ytr, yte = self.y[train_index], self.y[test_index]
 
            x_tr.append(xtr)
            x_te.append(xte)
            y_tr.append(ytr)
            y_te.append(yte)
 
        return x_tr, x_te, y_tr, y_te #均为列表
    
    
class PLSCV(baseCV):        
    def __init__(self, x, y):
        baseCV.__init__(self, x, y)

 
    
    def cv_predict(self, n_folds, max_components):
        
                     
        x_tr, x_te, y_tr, y_te = baseCV.CV(self, self.x, self.y, n_folds)
        
#        print x_train, x_test, y_train, y_test
        
#        print np.shape(y_predict)
        y_predict_all=np.ones((1,max_components))
#        print np.shape(Y_predict)
        pls = _NIPALS(max_components)
        for i in range(n_folds):
            
            y_predict = np.zeros((x_te[i].shape[0],max_components))
            xtrainmean = np.mean(x_tr[i], axis=0)
            ytrainmean = np.mean(y_tr[i], axis=0)

            xte_center = np.subtract(x_te[i], xtrainmean)                              
            yte_center = np.subtract(y_te[i],ytrainmean)
   
            w,T,P,lists_coefs_B=pls.fit(x_tr[i],y_tr[i],max_components)
#            print lists_coefs_B
            for j in range(max_components):
               
                    
                y_pre_center = np.dot(xte_center,lists_coefs_B[j])
                Y_pre = y_pre_center + ytrainmean
#                print np.shape(Y_pre)
                
                y_predict[:,j]=Y_pre.ravel() 
                
                
            
            y_predict_all=np.vstack((y_predict_all,y_predict))
            
        y_predict_all=y_predict_all[1:]

#        print np.shape(y_predict_all)
#        print np.shape(self.y)
            
#        print self.y    
        return   y_predict_all,self.y
    
    def cv_mse(self,Y_predict_all,y):
       
        
        press=np.square(np.subtract(Y_predict_all,y))

        PRESS_all=np.sum(press,axis=0)
        
        RMSECV_array=np.sqrt(PRESS_all/self.n)
        min_RMSECV=min(RMSECV_array)
        comp_array=RMSECV_array.argsort()
#        print comp_array
        comp_best=comp_array[0]+1
#        print comp_best
#         comp_best=RMSECV.index(min_RMSECV)+1
#         print comp_best
        
          

        return  RMSECV_array,min_RMSECV,comp_best 
        
        
        






if __name__ == '__main__':
    import numpy as np
    from scipy import linalg
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    
#     fname = loadmat('TCM.mat')
#     D = fname
#     print D.keys()
#                    
#     x_train = D['Xtrn'][:]
# #    print np.shape(x_train)
#     y_train = D['ytrn'][:,0:1]
#                         
#     x_test = D['Xtst'][:]
#     y_test = D['ytst'][:,0:1]
#     x=np.vstack((x_train,x_test))
#     y=np.vstack((y_train,y_test))


#     fname = loadmat('OrangeJuiceRevise.mat')
#     D = fname
#     x_train = D['XTrn'][:]
#     y_train = D['YTrn'][:,0:1]
#     x_test = D['XTst'][:]
#     y_test = D['YTst'][:,0:1]
#     print y_test
    

#     

    
#     fname = loadmat('milk.mat')
#     D = fname
#     print D.keys()
#                   
#     x = D['X']
#     print np.shape(x)   
#     y = D['y'][:,0:1]
    
#     
#     fname = loadmat('NIRgrass.mat')
#     D = fname
#     print D.keys()
#              
#     x = D['specgrass']
#     print np.shape(x)
#     y = D['propgrass'][:,0:1]
    
#     
    fname = loadmat('NIRcorn.mat')
    D = fname
    print D.keys()
             
    x = D['cornspect']
    print np.shape(x)
#    
    y = D['cornprop'][:,0:1]
 
    
#     fname = loadmat('NIRtablet.mat')
#     D = fname
# #    print D.keys()
# #           
#     x = D['spectablet'][:]
#     print np.shape(x)
#     y = D['proptablet'][:,0:1]
        
#    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
# 
#     fname = loadmat('nirbeer.mat')
#     D = fname
#     print D.keys()
#                     
#     x_train = D['Xcal'][:]
#     y_train = D['ycal'][:,0:1]
#           
#                   
#     x_test = D['Xtest'][:]
#     y_test = D['ytest'][:,0:1]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
# 
    max_components=20 
    n_folds=10
    
    plscv=PLSCV(x_train,y_train)
    Y_predict_all,y_test=plscv.cv_predict(n_folds, max_components)
    RMSECV_array,min_RMSECV,comp_best =plscv.cv_mse(Y_predict_all,y_test)
#    RMSECV1,min_RMSECV1,comp_best1 =plscv.cv_mse(Y_predict_all,y)
#     print RMSECV
#     print min_RMSECV
    print comp_best
#     print comp_best, RMSECV