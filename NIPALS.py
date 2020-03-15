# coding=utf-8
'''
Created on 2015年12月9日

@author: lenovo
'''


from __future__ import division
import warnings
import numpy as np
from scipy import linalg




def _nipals_twoblocks_inner_loop(X, Y, max_iter=500, tol=1e-06,):  # 閸愬懘鍎村顏嗗箚鐠侊紕鐣粁_weights
    """Inner loop of the iterative NIPALS algorithm.

    Provides an alternative to the svd(X'Y); returns the first left and rigth
    singular vectors of X'Y.  See PLS for the meaning of the parameters.  It is
    similar to the Power method for determining the eigenvectors and
    eigenvalues of a X'Y.
    """
    y_score = Y[:, [0]] 
    x_weights_old = 0
    ite = 1

    while True:
        # 1.1 Update u: the X weights
        # regress each X column on y_score
        x_weights = np.dot(X.T, y_score) / np.dot(y_score.T, y_score)  # w=X.T*Y[:,0]/||Y[:,0]||
        # 1.2 Normalize u
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights))  # w=w/||w||
        # 1.3 Update x_score: the X latent scores
        x_score = np.dot(X, x_weights)  # t=X*w
        # 2.1  regress each Y column on x_score
        y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)  # q=Y*t/(t.T*t)

        # 2.2 Update y_score: the Y latent scores
        y_score = np.dot(Y, y_weights) / np.dot(y_weights.T, y_weights)  # u=Y*q/(q.T,q)

        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol :
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached')
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights  # 瀵版鍩孹閻ㄥ嫭娼堥柌宄�Y閻ㄥ嫭娼堥柌锟�


def _center_xy(X, Y):  # 鐎佃鏆熼幑顔跨箻鐞涘奔鑵戣箛鍐ㄥ婢跺嫮鎮婇敍宀冪箲閸ョ偛顦甸悶鍡楁倵閻ㄥ嫭鏆熼崐鐓庢嫲閸у洤锟�
    """ Center X, Y and scale if the scale parameter==True
    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X_center = np.subtract(X, x_mean)
    y_mean = Y.mean(axis=0)
    Y_center = np.subtract(Y, y_mean)

    return X_center, Y_center, x_mean, y_mean



class _NIPALS():      
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm, constructors' parameters
    allow to obtain a specific implementation such as:

    
    This implementation uses the PLS Wold 2 blocks algorithm based on two
    nested loops:
        (i) The outer loop iterate over components.
        (ii) The inner loop estimates the weights vectors. This can be done
        with two algo. (a) the inner loop of the original NIPALS algo. or (b) a
        SVD on residuals cross-covariance matrices.

    Parameters
    ----------
    X : array-like of predictors, shape = [n_samples, p]
        Training vectors, where n_samples in the number of samples and
        p is the number of predictors.

    Y : array-like of response, shape = [n_samples, q]
        Training vectors, where n_samples in the number of samples and
        q is the number of response variables.

    n_components : int, number of components to keep. (default 2).

    scale : boolean, scale data? (default True)

    deflation_mode : str, "canonical" or "regression". See notes.

    mode : "A" classical PLS and "B" CCA. See notes.

    norm_y_weights: boolean, normalize Y weights to one? (default False)

    algorithm : string, "nipals" or "svd"
        The algorithm used to estimate the weights. It will be called
        n_components times, i.e. once for each iteration of the outer loop.

    max_iter : an integer, the maximum number of iterations (default 500)
        of the NIPALS inner loop (used only if algorithm="nipals")

    tol : non-negative real, default 1e-06
        The tolerance used in the iterative algorithm.

    copy : boolean
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effects.

    Attributes
    ----------
    `x_weights_` : array, [p, n_components]
        X block weights vectors.

    `y_weights_` : array, [q, n_components]
        Y block weights vectors.

    `x_loadings_` : array, [p, n_components]
        X block loadings vectors.

    `y_loadings_` : array, [q, n_components]
        Y block loadings vectors.

    `x_scores_` : array, [n_samples, n_components]
        X scores.

    `y_scores_` : array, [n_samples, n_components]
        Y scores.

    `x_rotations_` : array, [p, n_components]
        X block to latents rotations.

    `y_rotations_` : array, [q, n_components]
        Y block to latents rotations.

    coefs: array, [p, q]
        The coefficients of the linear model: Y = X coefs + Err

   
    """

    def __init__(self, n_components,  # 閸掓繂顬婇崠鏍у棘閺侊拷
                 max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X, Y, n_components):  # 閹风喎鎮庨弫鐗堝祦閿涘苯缂撶粩瀣侀崹瀣剁礉濮瑰倽袙閸欏倹鏆�

        
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        if n != Y.shape[0]:  # 娣囨繆鐦塜閿涘閻ㄥ嫭鐗遍張顑块嚋閺佹壆娴夐崥锟�            raise ValueError(
                'Incompatible shapes: X has %s samples, while Y '
                'has %s' % (X.shape[0], Y.shape[0])
        
        if self.n_components < 1 or self.n_components > p:  # 閹惰棄褰囬惃鍕瘜閹存劕鍨庨弫棰佺瑝閼宠棄鐨禍锟介敍灞肩瑝閼冲�绉存潻鍥у綁闁插繑鏆�
            raise ValueError('invalid number of components')

        Xcenter, Ycenter, self.x_mean_, self.y_mean_ = _center_xy(X, Y)  # 鏉堟挸鍙嗛弫鐗堝祦娑擃厼绺鹃崠鏍ь樀閻烇拷
        # Residuals (deflated) matrices
        Xk = Xcenter
        Yk = Ycenter
        # Results matrices
        self.x_scores_ = np.zeros((n, self.n_components))  
        self.y_scores_ = np.zeros((n, self.n_components))  
        self.x_weights_ = np.zeros((p, self.n_components)) 
        self.y_weights_ = np.zeros((q, self.n_components)) 
        self.x_loadings_ = np.zeros((p, self.n_components)) 
        self.y_loadings_ = np.zeros((q, self.n_components))

        
        # NIPALS algo: outer loop, over components
        for k in range(self.n_components):
            # 1) weights estimation (inner loop)
            # -----------------------------------

            x_weights, y_weights = _nipals_twoblocks_inner_loop(
                    X=Xk, Y=Yk, max_iter=self.max_iter,
                    tol=self.tol,)
#             elif self.algorithm == "svd":
#                 x_weights, y_weights = _svd_cross_product(X=Xk, Y=Yk)
            # compute scores
            x_scores = np.dot(Xk, x_weights)

            y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss
            
            
            # test for null variance
#             if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps:
#                 warnings.warn('X scores are null at iteration %s' % k)

            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            # - substract rank-one approximations to obtain remainder matrix
            Xk -= np.dot(x_scores, x_loadings.T)
            
#            print Xk

            y_loadings = (np.dot(Yk.T, x_scores)
                              / np.dot(x_scores.T, x_scores))
            Yk -= np.dot(x_scores, y_loadings.T)
            
#            print Yk
            
            

            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C 
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q


        lists_coefs = []              
        for i in range(n_components):
             
            self.x_rotations_ = np.dot(self.x_weights_[:, :i + 1], linalg.inv(np.dot(self.x_loadings_[:, :i + 1].T, self.x_weights_[:, :i + 1])))
            self.coefs = np.dot(self.x_rotations_, self.y_loadings_[:, :i + 1].T)
             
            lists_coefs.append(self.coefs)
        
#        print self.x_scores_

            
        return self.x_weights_, self.x_scores_, self.x_loadings_, lists_coefs


    def predict(self, x_test, coefs_B, xtr_mean, ytr_mean):  # 鐏忓棗澧犻棃銏㈡畱瀵版鍩岄惃鍕棘閺侀绱舵潻娑欐降,鐠侊紕鐣荤拠顖氭▕
        """
     
 
        Parameters
        ----------
        X : array-like of predictors, shape = [n_samples, p]
            Training vectors, where n_samples in the number of samples and
            p is the number of predictors.
 
         
        """
 
        xte_center = np.subtract(x_test, xtr_mean)
        y_pre = np.dot(xte_center, coefs_B)
        y_predict = np.add(y_pre, ytr_mean)
         
         
        return y_predict
    
    
    
    
if __name__ == '__main__':
    import numpy as np
    from scipy import linalg
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split   
    fname = loadmat('NIRcorn.mat')
    D = fname
    print D.keys()
    
    x = D['cornspect']
#    print np.shape(x)

    y = D['cornprop'][:, 3:4]

    n_components = 15
 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    pls = _NIPALS(n_components)
    
    xtr_mean = x_train.mean(axis=0)
    ytr_mean = y_train.mean(axis=0)  
    xte_center_ = np.subtract(x_test, xtr_mean) 
    W, T, P, lists_coefs_B = pls.fit(x_train, y_train, n_components) 
#     print lists_coefs_B
#     print np.shape(lists_coefs_B)
#     print np.shape(x_train)
#     print np.shape(y_train)
    
    y_predict = pls.predict(x_test, lists_coefs_B[5], xtr_mean, ytr_mean)
#    print np.shape(y_predict)
#     print y_predict
    rmse = np.sqrt(np.mean(np.square(y_predict - y_test), axis=0))
#     plt.plot(y_test,y_predict,'bo',label='test')
#     plt.plot(y_test,y_test)
#     plt.legend()
#     plt.show()
