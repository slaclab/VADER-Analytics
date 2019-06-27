import numpy as np
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import preprocessing

class SupportVectorRegression:

    def __init__(self,featureSelection=True,adaptive=False):
        np.set_printoptions(precision=5)
        #self.model=SVR(kernel='rbf', C=1e3, gamma=0.1,max_iter=4000)
        self.model = SVR(kernel='rbf', max_iter=4000)
        self.selectionModel=None
        self.featureSelection=featureSelection
        self.adaptive=adaptive
        # normalization data
        self.min_max_scaler = preprocessing.MinMaxScaler()

    def generate_x(self, X_in, dates, stepAhead):
        return X_in

    def fit(self, X, y, val_ratio=0.0):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X)


    def getRMSE(self,y,y_predict):
        return np.sqrt((np.power(y_predict-y,2)).sum()/y_predict.size)

    def fit(self,X,y):
        X=self.min_max_scaler.fit_transform(X)
        if len(y.shape)<=1:
            y=y.reshape([-1,1])
        yhat=None
        if self.featureSelection:
            self.selectionModel=SelectKBest(f_regression, k=8)
            X = self.selectionModel.fit_transform(X, y.ravel())
        self.model.fit(X,y.ravel())
        self.x_train=X
        self.y_train=y

    def predict(self,X,y):
        if len(X.shape)<=1:
            X=X.reshape([1,-1])
        X = self.min_max_scaler.transform(X)
        if len(y.shape)<=1:
            y=y.reshape([-1,1])
        if self.featureSelection:
            X=self.selectionModel.transform(X)
        if self.adaptive:
            print 'adaptive...'
            return self.predict_adaptive2(X,y)
        else:
            yhat=self.model.predict(X)
            yhat=np.array(yhat)
            score=self.getRMSE(y, yhat)
            return yhat,score

    def predict_adaptive2(self,X,y):
        yhat=np.empty((X.shape[0],y.shape[1]))
        step=180#24
        count=X.shape[0]/step
        for i in xrange(count):
            _x=X[i*step:(i+1)*step,:]
            yhat[i*step:(i+1)*step,:]=self.model.predict(_x).reshape([-1,1])
            _y=y[i*step:(i+1)*step,:]
            #self.model.partial_fit(_x,_y,steps=1)
            self.x_train=np.concatenate([self.x_train[1:,:],_x])
            self.y_train=np.concatenate([self.y_train[1:,:],_y])
            self.model.fit(self.x_train,self.y_train.ravel())

        return yhat,self.getRMSE(y,yhat)

    def predict_adaptive(self,X,y):
        yhat=np.empty((X.shape[0],y.shape[1]))
        for i in xrange(X.shape[0]):
            _x=np.expand_dims(X[i,:],axis=0)
            yhat[i,:]=self.model.predict(_x)
            _y=y[i,:].reshape([1,-1])
            self.x_train=np.concatenate([self.x_train,_x])
            self.y_train=np.concatenate([self.y_train,_y])
            self.model.fit(self.x_train,self.y_train.ravel())
            #print 'i=',i
        return yhat,self.getRMSE(y,yhat)
