# -*- coding: utf-8 -*-
'''
Created on 2018��1��20��

@author: Administrator
'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.io.matlab.mio import loadmat
def test_predict(y_test, AT_pre, SBC_pre, PDS_pre):

    plt.scatter(y_test, AT_pre, s=70, c='r', marker='o')
    plt.scatter(y_test, SBC_pre, s=70, c='b', marker='>')
    plt.scatter(y_test, PDS_pre, s=70, c='g', marker='*')
        
#     plt.plot(y_test,y_predict,'r^')     #蓝色圆圈
#     plt.plot(new_test,new_pre,'bo')     #红色三角形
#     plt.plot(SBC_test,SBC_pre,'m*')      #洋红色*
#     plt.scatter(MSC_test, MSC_pre,s=45,c='r',marker='>')
#     plt.plot(MSC_test,MSC_pre,'b*')
#     plt.plot(PDS_test,PDS_pre,'ch')     #青色六边形
   
    plt.xlabel('Measure value')
    plt.ylabel('predict value')
    plt.legend(('CT_PLS', 'SBC', 'PDS'), 'upper left')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'black', label='test')
#     plt.legend()
    plt.show()
    
def line_rmsecv_comp(Max_Compment, new_RMSECV):
# 3 [ 0.36874847  0.37740701  0.40410961  0.42428475  0.38769967  0.42984315 0.47497275  0.48855329  0.38599293  0.35320643  0.50539979  0.3652353 0.45962091  0.37383731  0.48056009]
# 0 [ 0.26070188  0.2321409   0.24968765  0.23019264  0.24309134  0.23048402  0.25190858  0.2602104   0.23963425  0.25019901  0.25170047  0.24178507  0.22790251  0.27937849]
# 1 [ 0.06922356  0.0755736   0.07422561  0.07560496  0.07188979  0.09194415 0.08145837  0.07141555  0.07641886  0.08159781  0.10054262  0.07129425  0.07258367  0.07744153  0.06984705]
# 2 [ 0.11368633  0.16760588  0.16250312  0.16689755  0.162887    0.18742581  0.18095945  0.21027667  0.158576    0.19266619  0.15703763  0.20896907  0.23643935  0.25332188  0.21569128]
    
#     plt.plot(range(1, Max_Compment + 1), SBC_RMSECV, '-o', color='red')
#     plt.plot(range(1, Max_Compment + 1), PDS_RMSECV, '-o', color='green')
    plt.plot(range(1, Max_Compment + 1), new_RMSECV, '-o', color='blue')
    
    plt.xlabel('number of Component')
    plt.ylabel('RMSECV')
    sns.set(context='paper', style='white')
#     plt.legend(('PLS-RMSECV'), 'upper left')
    sns.despine()
    plt.show()
    
def line_rmsep_stdNum(pls_rmsep, psbct_rmsep):
    X = [5, 8, 11, 14, 17, 20, 23, 26, 29]
#     plt.title("m5spec--mp6spec")
    plt.plot(X, pls_rmsep, '--', color='blue')
    plt.plot(X, psbct_rmsep, '-o', color='red')   
    plt.xticks(range(5, 30, 3))
    
    plt.xlabel('number of standard sample')
    plt.ylabel('RMSEP')
    plt.legend(('pls_rmsep', 'psbct_rmsep', 'new_psbct_rmsep'), 'upper right')
    plt.show()

def line_rmsep_stdNum1(pls_rmsep, psbct_rmsep):
    X = [5, 8, 11, 14, 17, 20, 23, 26, 29]
#     plt.title("m5spec--mp6spec")
    plt.plot(X, pls_rmsep, '--', color='blue')
    plt.plot(X, psbct_rmsep, '-o', color='red')   
    plt.xticks(range(5, 30, 3))
    
    plt.xlabel('number of standard sample')
    plt.ylabel('RMSEP')
    plt.legend(('SBC_rmsep', 'SBC_affine_rmsep'), 'upper right')
    plt.show()   
    
def spectrum(X):    
    wavelength = np.arange(1100, 2500, 2)
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength, X.T)
#     plt.legend([str], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.show()   

def Y_Ypre(ym, ys):
    n1, m1 = np.shape(ym)
    n2, m2 = np.shape(ys)

    X1 = [x for x in range(0, n1)]
    X2 = [x for x in range(0, n2)]
    
    plt.plot(X1, ym, '-o', color='blue')
    plt.plot(X2, ys, '-o', color='red')
    plt.xticks(range(0, n1 + 1, 1))
    plt.xlabel('sample_num')
    plt.ylabel('y')
    plt.legend(('y', 'y_pre'), 'upper left')
    plt.show()
    
def line_fit(T, y_m, y_s, T_m_std, T_s_std, y_m_pre, y_s_pre):

    plt.plot(T_m_std, y_m_pre, 'o', color='blue')
    plt.plot(T_s_std, y_s_pre, 'o', color='red')
    plt.plot(T, y_m, '-', color='blue')
    plt.plot(T, y_s, '-', color='red')
    plt.xticks(range(-1, 3, 1))

    plt.yticks(range(-1, 3, 1))
    
    plt.xlabel('T')
    plt.ylabel('y')
    plt.show()
    
def line_fit1(Tm, Ts, y_m, y_s, T_m_std, T_s_std, y_m_pre, y_s_pre):

    plt.plot(T_m_std, y_m_pre, 'o', color='blue')
    plt.plot(T_s_std, y_s_pre, 'o', color='red')
    plt.plot(Tm, y_m, '-', color='blue')
    plt.plot(Ts, y_s, '-', color='red')
    
    plt.xlabel('T')
    plt.ylabel('y')
    plt.show()

def sample_compare(T_m_std, T_s_std, y_m_pre, y_s_pre):

    plt.plot(T_m_std, y_m_pre, 'o', color='blue')
    plt.plot(T_s_std, y_s_pre, 'o', color='red') 
    plt.xlabel('T')
    plt.ylabel('y')
    plt.show()
    
    
def Ni_he(ym, ys, y_true, y_pre):
    n1, m1 = np.shape(ym)
    n2, m2 = np.shape(ys)
    X1 = [x for x in range(0, n1)]
    X2 = [x for x in range(0, n2)]
    X = [0, n1 - 1]
#     print np.shape(X), np.shape(y_true), np.shape(y_pre)
    
    plt.plot(X1, ym, 'o', color='blue')
    plt.plot(X2, ys, 'o', color='red')
    plt.plot(X, y_true, '-', color='blue')
    plt.plot(X, y_pre, '-', color='red')
    
    plt.xticks(range(0, n1 + 1, 1))
    plt.xlabel('sample_num')
    plt.ylabel('y')
    plt.legend(('y', 'y_pre'), 'upper left')
    plt.show()

def zong_tu(ym, ys, y_true, y_yuan, y_pingYi, y_xuanZhuan):
    n1, m1 = np.shape(ym)

    X1 = [x for x in range(0, n1)]
    X2 = [x for x in range(0, n1)]
    X = [0, n1 - 1]
#     print np.shape(X), np.shape(y_true), np.shape(y_pre)
    
    plt.plot(X1, ym, 'o', color='blue')
    plt.plot(X2, ys, 'o', color='red')
    plt.plot(X, y_true, '-', color='blue')
    plt.plot(X, y_yuan, '-', color='red')
    plt.plot(X, y_pingYi, '--', color='red')
    plt.plot(X, y_xuanZhuan, '-o', color='green')
    
    plt.xticks(range(0, n1 + 1, 1))
    plt.xlabel('sample_num')
    plt.ylabel('y')
#     plt.legend(('y', 'y_pre'), 'upper left')
    plt.show()
    
def spectrumT(T):    
    wavelength = np.arange(10, 12, 2)
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength, T.T)
#     plt.legend(['m5spec'], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.show()
    
    # 用于画论文光谱图
def drowAllSpectrum(corn_master,corn_slave,yaopian_master,yaopian_slave):
    plt.figure(dpi = 80)
    
    
    plt.subplot(2,3,1)
    plt.title("A")
    wavelength1 = np.arange(1100, 2500, 2)
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength1, corn_master.T)
#     plt.legend([str], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
    
    plt.subplot(2,3,2)
    plt.title("B")
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength1, corn_slave.T)
#     plt.legend([str], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
    
    
    corn_test_master = corn_master[[40],:]
    corn_test_slave = corn_slave[[40],:]
    yaopian_test_master = yaopian_master[[125],:]
    yaopian_test_slave = yaopian_slave[[125],:]
    
    
    plt.subplot(2,3,3)
    plt.title("C")
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength1, corn_test_master.T,color="red")
    plt.plot(wavelength1, corn_test_slave.T,color='blue')
#     plt.legend([str], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
    
    
    
#     药片集不能用length1
    plt.subplot(2,3,4)
    plt.title("D")
    wavelength2 = np.arange(600, 1900, 2)
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength2, yaopian_master.T)
#     plt.legend([str], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
    
    plt.subplot(2,3,5)
    plt.title("E")
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength2, yaopian_slave.T)
#     plt.legend([str], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
    
    
#     从玉米集中取出第41个样本，从药片集中取出第126个样本，作图
    
    
    
    
    print corn_test_master.shape , corn_test_slave.shape , corn_master.shape
     
    
     
     
    plt.subplot(2,3,6)
    plt.title("F")
#     plt.plot(wavelength, X_new.T)
    plt.plot(wavelength2, yaopian_test_master.T , color='red')
    plt.plot(wavelength2, yaopian_test_slave.T , color='blue')
#     plt.legend([str], loc='upper center')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('Absorbance')
     
    plt.show()
      
    
    
    
if __name__ == "__main__":
    fname = loadmat('NIRcorn.mat')
    X_corn_master = fname['m5spec']['data'][0][0]
    
    
    X_corn_slave = fname['mp6spec']['data'][0][0]
    
    
    
    fname1 = loadmat('Pharmaceutical tablet.mat')
    
    X_yaopian_master = fname1['test_1']['data'][0][0]
    
    
    X_yaopian_slave = fname1['test_2']['data'][0][0]
    
    
    drowAllSpectrum( X_corn_master , X_corn_slave , X_yaopian_master , X_yaopian_slave )
    
#     spectrum(X_master)
