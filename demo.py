# -*- coding: utf-8 -*-
'''
Created on 2018��8��9��
调用各种核方法
@author: Zzh
'''
from TCA import TCA
from scipy.io.matlab.mio import loadmat
from sklearn.cross_validation import train_test_split
from KS_algorithm import KennardStone
from function_module import SNV

class datasetProcess:
    '''
            导入不同的数据集，将数据集对应的参数返回      
    '''
    def __init__(self, dataType=0 , bool = 0):
        # dataType : 数据集的类型,不同的数字代表不同的数据集
        self.dataType = dataType
        self.bool = bool
    
    def datasetImport(self):
        ''' 
        param: None
        return：
            X_master：主仪器光谱数据
            X_slave：从仪器光谱数据
            y：主从仪器光谱数据的响应值
        '''
        fname1 = loadmat('NIRcorn.mat')
        fname2 = loadmat('wheat_B_cal.mat')
        
        fname3 = loadmat('Pharmaceutical tablet.mat')
        fname4 = loadmat("wheat_A_cal.mat")
        if self.dataType < 4:
            print 'cornprop--y:', self.dataType
            X_master = fname1['mp6spec']['data'][0][0]
            y = fname1['cornprop'][:, self.dataType:self.dataType + 1]
            X_slave = fname1['mp5spec']['data'][0][0]    
        if self.dataType == 4:
            print 'B1--B2'
            X_master = fname2['CalSetB1']
            X_slave = fname2['CalSetB2']
            y = fname2['protein']
        if self.dataType == 5:
            print 'B1--B3'
            X_master = fname2['CalSetB1']
            X_slave = fname2['CalSetB3']
            y = fname2['protein']
        if self.dataType == 6:
            print 'B2--B3'
            X_master = fname2['CalSetB2']
            X_slave = fname2['CalSetB3']
            y = fname2['protein']
        if self.dataType == 7:   
            print 'tablet---0'
            X_master = fname3['test_1']['data'][0][0]
            X_slave = fname3['test_2']['data'][0][0]
            y = fname3['test_Y']['data'][0][0][:, 0:1]
        if self.dataType == 8:  
            print 'tablet---1' 
            X_master = fname3['test_1']['data'][0][0]
            X_slave = fname3['test_2']['data'][0][0]
            y = fname3['test_Y']['data'][0][0][:, 1:2]
        if self.dataType == 9:  
            print 'tablet---2' 
            X_master = fname3['test_1']['data'][0][0]
            X_slave = fname3['test_2']['data'][0][0]
            y = fname3['test_Y']['data'][0][0][:, 2:3]
        if self.dataType == 10:   
            print 'A2--A1'
            X_master = fname4['CalSetA2']
            y = fname4['protein']
            X_slave = fname4['CalSetA1']
        if self.dataType == 11:  
            print 'A3--A1' 
            X_master = fname4['CalSetA3']
            y = fname4['protein']
            X_slave = fname4['CalSetA1']
        if self.dataType == 12:  
            print 'A3--A2' 
            self.X_master = fname4['CalSetA3']
            self.y = fname4['protein']
            self.X_slave = fname4['CalSetA2']
#         X_master = SNV(X_master)
#         X_slava = SNV(X_slave)
        if self.dataType == 13:
            print 'tablet---all' 
            X_master = fname3['test_1']['data'][0][0]
            X_slave = fname3['test_2']['data'][0][0]
            y = fname3['test_Y']['data'][0][0][:, 0:3]
        if self.bool == 1:
            print "TCA_SNV_deal"
            X_master = SNV(X_master)
            X_slave = SNV(X_slave)
        return X_master, X_slave, y
    
    def datasetSplit_Random(self, X_master, X_slave, y, next_size=0.2, random_state=0):
        '''
                        将导入的数据集采用随机的方式进行划分
        Param：
            X_master：主仪器光谱数据
            X_slave：从仪器光谱数据
            y：    主从仪器光谱数据的响应值
            next_size: 后半部分所占的比例
            random_state: 随机为伪随机的方式，不同的随机率对应不同的划分
        return：
            X_master_prior：主仪器光谱数据划分后的前半部分
            X_master_next：主仪器光谱数据划分后的后半部分
            X_slave_prior：从仪器光谱数据划分后的前半部分
            X_slave_next：从仪器光谱数据划分后的后半部分
            y_prior：主从仪器光谱数据的响应值的前半部分
            y_next：主从仪器光谱数据的响应值的后半部分
        '''
        X_master_prior, X_master_next, y_prior, y_next = train_test_split(X_master, y, test_size=next_size, random_state=random_state)
        X_slave_prior, X_slave_next, y_prior, y_next = train_test_split(X_slave, y, test_size=next_size, random_state=random_state)

        return  X_master_prior, X_master_next, X_slave_prior, X_slave_next, y_prior, y_next
    def datasetSplit_KS(self, X_master, X_slave, y, num=5):
        '''
                        将导入的数据集采用KS算法进行划分,找到方差最大的样本
        Param：
            X_master：主仪器光谱数据
            X_slave：从仪器光谱数据
            y：    主从仪器光谱数据的响应值
            num: 前半部分抽取的方差最大的样本数目
        return：
            X_master_prior：主仪器光谱数据划分后的前半部分
            X_master_next：主仪器光谱数据划分后的后半部分
            X_slave_prior：从仪器光谱数据划分后的前半部分
            X_slave_next：从仪器光谱数据划分后的后半部分
            y_prior：主从仪器光谱数据的响应值的前半部分
            y_next：主从仪器光谱数据的响应值的后半部分
        '''
        KS_demo = KennardStone(X_master, num)
        print X_slave.shape , num
        CalInd_master, ValInd_master = KS_demo.KS() 
        # CalInd_master方差最大样本的下标 ValInd_master 剩余样本的下标
        X_master_prior = X_master[CalInd_master]
        X_master_next = X_master[ValInd_master]
             
        X_slave_prior = X_slave[CalInd_master]
        X_slave_next = X_slave[ValInd_master]
        
        y_prior = y[CalInd_master]
        y_next = y[ValInd_master]
        
        return X_master_prior, X_master_next, X_slave_prior, X_slave_next, y_prior, y_next        

if __name__ == '__main__':
    import numpy as np
    from tca_param_choose import tca_param_choose
    
    rmsep_list = []
    m_list = []
    for i in range(7):
        # 导入数据
        print i
        DP_demo = datasetProcess(dataType=i)
        X_master, X_slave, y = DP_demo.datasetImport()
        X_master_cal, X_master_test, X_slave_cal, X_slave_test, y_cal, y_test = DP_demo.datasetSplit_Random(X_master, X_slave, y, next_size=0.2)
        
        X_m_train, X_m_val, y_m_train, y_m_val = train_test_split(X_master_cal, y_cal, test_size=0.5, random_state=0)
        KS_demo = KennardStone(X_slave, 64)
        CalInd, ValInd = KS_demo.KS()  
        
        X_s = X_slave[CalInd]  # 从仪器有标签的样本
        X_s_o = X_slave[ValInd]  # 从仪器没有标签的样本
        y_s = y[CalInd]  # 从仪器样本的标签
        
        # 选择参数
        TPC_demo = tca_param_choose(X_m_train, y_m_train, X_m_val, y_m_val, X_s, y_s, X_s_o)
        m_op, k_op = TPC_demo.choose_Param(15)
        print m_op , "m_op"
        m_list.append(m_op)
        
        my_tca = TCA(dim=m_op)
        
        T_m_cal, T_s_o, T_slave_test, T_s = my_tca.fit_transform(X_master_cal, X_slave_cal, X_slave_cal, X_s)
        
    #     k = np.linalg.lstsq(T_m_cal, y_cal)[0]
    #     y_test_pre = np.dot(T_slave_test, k)
    #     RMSEP = np.sqrt(np.sum(np.square(np.subtract(y_test, y_test_pre)), axis=0) / y_test_pre.shape[0])
    #     print RMSEP
        
        T = np.vstack((T_m_cal, T_s))
        y = np.vstack((y_cal, y_s))
        k1 = np.linalg.lstsq(T, y)[0]
        print np.shape(k1), np.shape(T_slave_test)
        y_test_pre = np.dot(T_slave_test, k1)
        RMSEP = np.sqrt(np.sum(np.square(np.subtract(y_cal, y_test_pre)), axis=0) / y_test_pre.shape[0])
        print RMSEP , "rmsep"
        rmsep_list.append(RMSEP)
        
    print rmsep_list
    print m_list
    
    np.savetxt('rmsep.csv', rmsep_list, delimiter=',', fmt='%.6f')
    
    np.savetxt('m.csv', m_list, delimiter=',', fmt='%.6f')
    
    
    




