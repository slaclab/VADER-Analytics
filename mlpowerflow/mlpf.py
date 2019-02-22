# import mlpf -- contains all functions of machine learning powerflow project

import numpy as np
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

class forward_mlpf:
    
    # ML Powerflow Forward Mapping
    
    def __init__(self, numbus):
        self.numbus           = numbus # The total number of nodes in the network
        self.itworked         = 'Initialized!'
        self.p = np.zeros((1,1))
        self.q = np.zeros((1,1))
        self.v = np.zeros((1,1))
        self.a = np.zeros((1,1))
        self.uw = np.zeros((1,1))
        self.uw_norm = np.zeros((1,1))
        self.pq_norm = np.zeros((1,1))
        self.X_train = np.zeros((1,1))
        self.y_train = np.zeros((1,1))
        self.X_test = np.zeros((1,1))
        self.y_test = np.zeros((1,1))
        self.Ntrain = 0
        self.Ntest = 0
        self.N = 0 # Total number in data set
        self.n = 0
#         self.modelcounter     = 0   # Number of source signals
#         self.models           = {}  # Model for each source signal, this is a list but could be a dict
#         self.constraints      = []  # Additional constraints
#         self.N                = len(aggregateSignal) # Length of aggregate signal


    def supply_full_data(self, p, q, v, a, n, N):
        
        # Store the data
        
        self.p = np.copy(p)
        self.q = np.copy(q)
        self.v = np.copy(v)
        self.a = np.copy(a)
        
        # Cartesian coordinates
        u = v * np.cos(a) # Make sure a is in radians
        w = v * np.sin(a)
        uw = np.zeros((N,2*n))
        
        self.uw = np.copy(uw)
        self.N = N
        self.n = n
        
        
        
    def process_data(self,THRESHOLD):

        # Process the data ---- Replace with standardscaler
        uw_mean = np.mean(self.uw,axis=0)
        uw_std = np.std(self.uw,axis=0)
        p_mean = np.mean(self.p,axis=0)
        p_std = np.std(self.p,axis=0)
        q_mean = np.mean(self.q,axis=0)
        q_std = np.std(self.q,axis=0)
        
        N = self.N
        n = self.n
        uw_norm = np.zeros((N,2*n))
        pq_norm = np.zeros((N,2*n))
#         q_norm = np.zeros((N,n))
        for j in range(n):
            if uw_std[j] != 0:
                uw_norm[:,j] = ( self.uw[:,j] - uw_mean[j] ) / uw_std[j]
            else: 
                uw_norm[:,j] = ( self.uw[:,j] - uw_mean[j] )
            if p_std[j] != 0:
                pq_norm[:,j] = ( self.p[:,j] - p_mean[j] ) / p_std[j]
            else: 
                pq_norm[:,j] = ( self.p[:,j] - p_mean[j] )
            if q_std[j] != 0:
                pq_norm[:,j+n] = ( self.q[:,j] - q_mean[j] ) / q_std[j]
            else: 
                pq_norm[:,j+n] = ( self.q[:,j] - q_mean[j] )
            if uw_std[j+n] != 0:
                uw_norm[:,j+n] = ( self.uw[:,j+n] - uw_mean[j+n] ) / uw_std[j+n]
            else: 
                uw_norm[:,j+n] = ( self.uw[:,j+n] - uw_mean[j+n] )
            
        self.uw_norm = np.copy(uw_norm)
        self.pq_norm = np.copy(pq_norm)
#         self.q_norm = np.copy(q_norm)
        
#         # Only train on columns where the variability is still significant after normalization
#         training_cols = []
#         for col in range(n):
#             if abs(uw_mean[col]) > THRESHOLD:
#                 training_cols.append(col)
                
#         self.training_cols = training_cols
        
    def train_test_split_data(self, rand_percent = True, train_percent = 0.8, time_sers = False, time_percent = 0.8):
        
        if rand_percent == True:
            X_train, X_test, y_train, y_test = train_test_split(self.uw_norm, self.pq_norm, train_size=train_percent, random_state=42)
            Ntrain = np.shape(X_train)[0]
            Ntest = np.shape(X_test)[0]
        if time_sers == True:
            Ntrain = int(time_percent*np.shape(self.uw_norm)[0])
            Ntest = np.shape(self.uw_norm)[0] - Ntrain
            X_train = self.uw_norm[np.arange(0,Ntrain),:]
            X_test = self.uw_norm[np.arange(Ntrain,Ntrain+Ntest),:]
            y_train = self.pq_norm[np.arange(0,Ntrain),:]
            y_test = self.pq_norm[np.arange(Ntrain,Ntrain+Ntest),:]
            
        self.X_train = np.copy(X_train)
        self.X_test = np.copy(X_test)
        self.y_train = np.copy(y_train)
        self.y_test = np.copy(y_test)
        self.Ntrain = Ntrain
        self.Ntest = Ntest
    
        return
    
    
    def fit_svr(self, C_set, eps_set, maxiter):

        n = 2*self.n
        Ntrain = self.Ntrain

        b = np.zeros((n,1))
        coeffs = np.zeros((n,Ntrain))

        num_SV = np.zeros((n,1))
        SV_inds = np.zeros((n,Ntrain))

        C_best = np.zeros((n,1))
        eps_best = np.zeros((n,1))

        for j in range(n): 

            tuned_parameters = [{'kernel': ['poly'],'degree': [2.0], 'C': C_set, 'epsilon': eps_set,'max_iter':[maxiter]}]

            scaler_x = StandardScaler()
            scaler_x.fit(self.X_train)
            scaler_y = StandardScaler()
            scaler_y.fit(self.y_train[:,j].reshape((self.Ntrain,1)))

            regr_test = GridSearchCV(SVR(),tuned_parameters)
            regr_test.fit(scaler_x.transform(self.X_train), (scaler_y.transform(self.y_train[:,j].reshape((self.Ntrain,1)))).reshape((self.Ntrain,)))

            regr_svr = regr_test.best_estimator_
            eps_best[j] = regr_svr.epsilon
            C_best[j] = regr_svr.C

            b[j] = regr_svr.intercept_
            num_SV[j,0] = np.size(regr_svr.support_)

            for i in np.arange(0,np.size(regr_svr.support_)):
                vals = regr_svr.dual_coef_
                coeffs[j,regr_svr.support_[i]] = vals.item(i) 
                SV_inds[j,regr_svr.support_[i]] = 1 
                
            self.b = b
            self.coeffs = coeffs
            self.num_SV = num_SV
            self.SV_inds = SV_inds
            self.C_best = C_best
            self.eps_best = eps_best
            self.regressor = regr_svr
            
            
    def apply_svr(self, X_sample):

        y_output = np.zeros((2*self.n,))
        scaler_x = StandardScaler()
        scaler_x.fit(self.X_train)        

        for k in range(2*self.n):
            K = polynomial_kernel(scaler_x.transform(self.X_train), scaler_x.transform(X_sample.reshape((1,2*self.n))),degree=2,gamma=1.0,coef0=1.0)
            scaler_y = StandardScaler()
            scaler_y.fit(self.y_train[:,k].reshape((self.Ntrain,1)))
            y_output[k] = scaler_y.inverse_transform((np.mat(self.coeffs[k,:])*np.mat(K)+self.b[k,0]).reshape((1,1)))

        return y_output


    def test_error_svr(self):

        test_error_vals = np.zeros((self.Ntest,1))
        test_y_vals = np.zeros((self.Ntest,2*self.n))

        for i in range(self.Ntest):

            test_y_vals[i,:] = forward_mlpf.apply_svr(self,self.X_test[i,:])
            test_error_vals[i] = np.linalg.norm(test_y_vals[i,:]-self.y_test[i,:],2)/np.power(self.n,0.5)

        total_rmse = np.sqrt(np.mean(np.power(test_y_vals - self.y_test,2)))
        
        self.test_y_vals = test_y_vals
        self.test_error_vals = test_error_vals
        self.total_rmse = total_rmse
