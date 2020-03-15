# -*- coding: utf-8 -*-
'''
Created on 2018��8��9��

@author: Zzh
'''
from TCA import TCA
import numpy as np

class tca_param_choose:
    def __init__(self, X_m_cal, y_m_cal, X_m_val, y_m_val, X_s, y_s, X_s_o):
        '''主仪器数据集分成两部份：一部分和从仪器数据集建立公共子空间(X_m_val)，另一部分进行参数选择(X_m_cal)
            从仪器数据集分成两部分，有标签的进行参数选择，无标签的那部分构建子空间
        '''
        self.X_m_cal = X_m_cal
        self.y_m_cal = y_m_cal
        self.X_m_val = X_m_val
        self.y_m_val = y_m_val
        self.X_s = X_s
        self.y_s = y_s
        self.X_s_o = X_s_o
        
    def choose_Param(self, max_m):
        '''确定最优的子空间维度
        param: max_m    最大的子空间维度
        return: m_op    最优的子空间维度
        k_op 最优子空间模型对应的最小二乘系数
        '''
        error_list = []
        k_list = []
        for m in range(1, max_m + 1):
            my_tca = TCA(dim=m)
            T_m_val, T_s_o, T_s, T_m_cal = my_tca.fit_transform(self.X_m_val, self.X_s_o, self.X_s, self.X_m_cal)         
#             print np.shape(T_m_val), np.shape(self.y_m_val)
            
            k = np.linalg.lstsq(T_m_val, self.y_m_val)[0]
            k_list.append(k)
            T = np.vstack((T_m_cal, T_s))
            
            y = np.vstack((self.y_m_cal, self.y_s))           
            y_pre = np.dot(T, k)
            
            error = self.rmse(y, y_pre)
            error_list.append(error)
          
        index_value = error_list.index(min(error_list))  
        m_op = index_value + 1
        k_op = k_list[index_value]
        
#         print error_list
        return m_op, k_op
    
         
    def rmse(self, y, y_pre):
        '''
        param: y，真实值 
                y_pre,预测值
        return： rmse 真实值和预测值之间的均方误差
        '''
        press = np.square(np.subtract(y, y_pre))
        press_sum = np.sum(press, axis=0)
        rmse = np.sqrt(press_sum / y.shape[0])
        
        return rmse
    
    
