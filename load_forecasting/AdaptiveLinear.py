import numpy as np
from sklearn import linear_model
from skimage.util.shape import view_as_windows

class AdaptiveLinear:

    def __init__(self, y):
        np.set_printoptions(precision=5)
        self.model = linear_model.LinearRegression()
        if len(y.shape) <= 1:
            y = y.reshape([-1, 1])
        self.y = y
        self.y_train = None
        self.x_train = None
        self.firstFit=True

    def getRMSE(self, y, y_predict):
        return np.sqrt((np.power(y_predict - y, 2)).sum() / y_predict.size)

    def generate_x(self, X_in, dates, stepAhead):
        xLoad = X_in[:, 168:]
        xTemp = X_in[:, :168]
        sizeTest=30*24
        lags = np.array([8, 12, 24])
        sizeVal = int(sizeTest / 2)
        bestLag = lags[0]
        bestScore = np.inf
        nTrain = xLoad.shape[0] - (sizeTest + sizeVal)
        for lag in lags:
            self.y_train=None
            self.firstFit = True
            X=xLoad[:,-lag:]
            self.x_train = X[:nTrain, :]
            self.y_train = self.y[:nTrain]
            x_val = X[nTrain:(nTrain + sizeVal), :]
            y_val = self.y[nTrain:(nTrain + sizeVal)]
            self.fit(self.x_train, self.y_train)
            score = self.getRMSE(y_val, self.model.predict(x_val))
            if bestScore > score:
                bestScore = score
                bestLag = lag
        X = xLoad[:, -bestLag:]
        self.firstFit = True
        return X

    def fit(self, X, y, val_ratio=0.0):
        self.model.fit(X, y.ravel())
        if self.firstFit:
            self.x_train=X
            if len(y.shape) <= 1:
                y = y.reshape([-1, 1])
            self.y_train=y
            nTrain = y.shape[0]
            self.firstFit=False

    def predict(self, X,y):
        if len(X.shape) <= 1:
            X = X.reshape([1, -1])
        if len(y.shape) <= 1:
            y = y.reshape([-1, 1])
        h_x = np.empty((X.shape[0], self.y.shape[1]))
        step = 48 # 24
        count = X.shape[0] / step
        for i in xrange(count):
            _x = X[i * step:(i + 1) * step, :]
            h_x[i * step:(i + 1) * step, :] = self.model.predict(_x).reshape([-1, 1])
            _y = y[i * step:(i + 1) * step, :]
            self.x_train = np.concatenate([self.x_train[1:, :], _x])
            self.y_train = np.concatenate([self.y_train[1:, :], _y])
            self.model.fit(self.x_train, self.y_train.ravel())
        '''
        for i in xrange(X.shape[0]):
            h_x[i, :] = self.model.predict(np.expand_dims(X[i, :], axis=0))
            self.x_train = np.concatenate([self.x_train, X[i, :].reshape([1, -1])])
            self.y_train = np.concatenate([self.y_train, y[i, :].reshape([-1, 1])])
            self.model.fit(self.x_train, self.y_train)
        '''
        return h_x
