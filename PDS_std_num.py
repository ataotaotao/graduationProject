#-*-coding:utf-8-*-
from function_module import *
from PDS_910 import PDS
def pds(init_width,max_width ,n_folds_cal , n_max_components_cal):
    
    RMSEP_list = []
    
    for i in range(5 , 31 , 1):
        print i
        dataImport = Dataset_Import(type = 0, std_num = 5 , bool = 1)
        X_master , X_slave , y = dataImport.dataset_return()
        num = int(X_master.shape[0]*0.8)
        
        X_m_cal, y_m_cal, X_s_cal, y_s_cal, X_s_test, y_s_test, X_m_test, y_m_test , X_m_std , y_m_std , X_s_std , y_s_std = Dataset_KS_split_std(X_master , X_slave , y , num, std_num = i )
        
        max_folds = 10
        max_components = 15
        
        pds=PDS(X_m_cal,X_s_cal,y_m_cal,X_m_std,X_s_std,X_s_test,y_s_test,init_width,max_width)
    #         print n_folds_cal , self.max_components , "n_fold_cal_max_components"
        Trans_matrix_list=pds.transform3(n_folds_cal,n_max_components_cal)
        best_width,best_trans_matrix,coefs_B_cal,err_list, RMSECV_list,comp_best=pds.cv_window(Trans_matrix_list,max_folds , max_components)
        print best_width , "best_width"
        y_predict,RMSEP=pds.predict(best_trans_matrix, coefs_B_cal,X_s_test,y_s_test)
        ycal_predict,RMSEC=pds.predict(best_trans_matrix, coefs_B_cal,X_s_cal,y_s_cal)
        RMSEP_list.append(RMSEP)
        print RMSEP , "RMSECP"
#         best_width, best_trans_matrix, coefs_B_cal, err_list, RMSECV_cal, comp_best_cal = pds.cv_window(Trans_matrix_list, n_folds_cal, max_components_cal)
    print RMSEP_list
    
        


if __name__ == "__main__":
    
    pds(3 , 16 , 5 , 3)




