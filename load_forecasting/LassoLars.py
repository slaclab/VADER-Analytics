import numpy as np
from sklearn import linear_model


class LassoLarsIC:
    def __init__(self):
        self.model = linear_model.LassoLarsIC(criterion='bic')

    def generate_x(self, X, dates, stepAhead):
        xLoad = X[:, 168:]
        xTemp = X[:, :168]
        newX = np.empty((xLoad.shape[0], 13))
        newX[:, 0] = dates[:].hour
        newX[:, 1] = dates[:].dayofweek
        newX[:, 2:6] = xTemp[:, -4:]  # last 4 hours available
        newX[:, 6:8] = xLoad[:, -2:]  # 1 and 2 hour lagged load (last 2 hours available)
        newX[:, 8] = np.mean(xLoad[:, -24:], axis=1)  # (average of last 24 hours available)
        newX[:, 9] = xLoad[:, -24 + stepAhead - 1]  # 24 hour lagged load (matching with output)
        newX[:, 10] = xLoad[:, -48 + stepAhead - 1]  # 48 hour lagged load (matching with output)
        newX[:, 11] = xLoad[:, -72 + stepAhead - 1]  # 72 hour lagged load (matching with output)
        newX[:, 12] = xLoad[:, -168 + stepAhead - 1]  # 168 hour lagged load (previous week - (matching with output))
        return newX

    def fit(self, X, y, val_ratio=0.0):
        # val_ratio = alpha
        # (1-alpha)=train and alpha=validation
        self.model.fit(X, y.ravel())

    def predict(self, X):
        return self.model.predict(X)
