import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class GradientBoosting:

    def __init__(self, validate=True,adaptive=False):
        self.x_train=None
        self.y_train = None
        np.set_printoptions(precision=5)
        self.model = GradientBoostingRegressor(loss='ls',n_estimators=100)
        self.adaptive=adaptive
        self.validate=validate

    def generate_x(self,X,dates,stepAhead):
        xLoad=X[:,168:]
        xTemp=X[:,:168]
        newX=np.empty((xLoad.shape[0],13))
        newX[:,0]=dates[:].hour
        newX[:,1]=dates[:].dayofweek
        newX[:,2:6]=xTemp[:,-4:]#last 4 hours available
        newX[:,6:8]=xLoad[:,-2:] # 1 and 2 hour lagged load (last 2 hours available)
        newX[:,8]=np.mean(xLoad[:,-24:],axis=1) #(average of last 24 hours available)
        newX[:,9]=xLoad[:,-24+stepAhead-1] # 24 hour lagged load (matching with output)
        newX[:,10]=xLoad[:,-48+stepAhead-1] # 48 hour lagged load (matching with output)
        newX[:,11]=xLoad[:,-72+stepAhead-1] # 72 hour lagged load (matching with output)
        newX[:,12]=xLoad[:,-168+stepAhead-1] # 168 hour lagged load (previous week - (matching with output))
        return newX

    def getRMSE(self, y, y_predict):
        return np.sqrt((np.power(y_predict-y,2)).sum()/y_predict.size)

    def fit(self, X, y):
        if len(y.shape)<=1:
            y=y.reshape([-1,1])
        if self.validate:
            return self.fit_validate(X,y)
        self.model.fit(X,y.ravel())
        self.x_train=X
        self.y_train=y

    def predict(self, X):
        if len(X.shape)<=1:
            X=X.reshape([1,-1])
        if self.adaptive:
            print 'adaptive...'
            return self.predict_adaptive(X)
        else:
            return self.model.predict(X)

    def predict_adaptive(self, X,y):
        yhat=np.empty((X.shape[0],1))
        for i in xrange(X.shape[0]):
            _x=np.expand_dims(X[i,:],axis=0)
            yhat[i,:]=self.model.predict(_x)
            _y=y[i,:].reshape([1,-1])
            self.x_train=np.concatenate([self.x_train,_x])
            self.y_train=np.concatenate([self.y_train,_y])
            self.model.fit(self.x_train,self.y_train.ravel())
        return yhat

    def fit_validate(self, X, y):
        n_estimators=np.array([1,20,50,100])
        bestN=0
        best_error=np.inf
        nVal=int(X.shape[0]*0.15)
        yVal=y[-nVal:]
        XVal = X[-nVal:,:]
        yTrain = y[:nVal]
        XTrain = X[:nVal,:]
        for i,n in enumerate(n_estimators):
            self.model = GradientBoostingRegressor(loss='ls',n_estimators=n)
            self.model.fit(XTrain,yTrain.ravel())
            y_val=self.model.predict(XVal)
            error_v=self.getRMSE(yVal,y_val)
            if error_v<best_error:
                best_error=error_v
                bestN=n

        self.model = GradientBoostingRegressor(loss='ls',n_estimators=bestN)
        self.model.fit(X, y.ravel())
        self.x_train = X
        self.y_train = y