
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')


class ForwardMLPF(object):
    """Build, train, implement, and test an ML model of the forward power flow equations.

    The model built in this class was published in the proceedings of NAPS 2017 as "Robust mapping rule estimation
    for power flow analysis in distribution grids" by Jiafan Yu, Yang Weng, and Ram Rajagopal. Please refer to the
    README.md for a link to the paper.

    The power flow functions define the map between voltages and power injections. In this implementation, a Support
    Vector Regression (SVR) model with a quadratic kernel is used to represent the mapping. All of the models
    parameters are trained directly from data taken at all of the buses in the network, without any direct knowledge of
    the line parameters or Ybus matrix.

    The methods in this v =  ingest and process voltage and power injection data, fit an SVR model for the mappings
    at each bus in the network, and test the mapping using a reserved test set.
    """

    def __init__(self, num_bus, num_samples):
        """Initialize attributes of the object."""
        
        self.num_bus = num_bus
        self.num_samples = num_samples
        self.p = np.zeros((1, 1))
        self.q = np.zeros((1, 1))
        self.v = np.zeros((1, 1))
        self.a = np.zeros((1, 1))
        self.uw = np.zeros((1, 1))
        self.pq = np.zeros((1, 1))
        self.num_train = 0
        self.num_test = 0
        self.X_train = np.zeros((1, 1))
        self.y_train = np.zeros((1, 1))
        self.X_test = np.zeros((1, 1))
        self.y_test = np.zeros((1, 1))
        self.test_y_values = np.zeros((1, 1))
        self.test_error_values = np.zeros((1, 1))
        self.total_rmse = np.zeros((1, 1))
        self.best_svr_models = {}
        self.b = np.zeros((1, 1))
        self.coeffs = np.zeros((1, 1))
        self.num_SV = np.zeros((1, 1))
        self.SV_inds = np.zeros((1, 1))
        self.C_best = np.zeros((1, 1))
        self.eps_best = np.zeros((1, 1))

    def supply_full_data(self, p, q, v, a, num_bus, num_samples):
        """
        Ingest the p, q, v, and a data into the object. Manipulate it into the input - output formats.

        The input to the ML model requires manipulating the voltage data, v and a, into cartesian form. The variables
        u and w define the cartesian version of the voltage data, and they are saved together in uw to be used as
        input to the model. The power variables p and q are combined and saved together in pq to be used as the output
        of the model.

        Parameters
        ----------
        p: array_like
            Real power injection in all buses for all samples
        q: array_like
            Reactive power injection in all buses for all samples
        v: array_like
            Voltage magnitude in all buses for all samples
        a: array_like
            Voltage phase angle in all buses for all samples
        num_bus: int
            Number of buses in the network
        num_samples: int
            Number of data samples provided

        Attributes
        ----------
        p, q, v, a: array_like
            Data as described above, saved as attributes.
        num_bus, num_samples: int
            Counts as described above, saved as attributes.
        uw: array_like
            Cartesian form of the voltage data
        pq: array_like
            Combined power injections
        """
        self.p = np.copy(p)
        self.q = np.copy(q)
        self.v = np.copy(v)
        self.a = np.deg2rad(np.copy(a))
        
        # Cartesian coordinates for input
        u = v * np.cos(a)  # Make sure a is in radians
        w = v * np.sin(a)
        uw = np.zeros((num_samples, 2*num_bus))
        uw[:, np.arange(0, num_bus)] = u
        uw[:, np.arange(num_bus, 2*num_bus)] = w
        # Joined p and q for output
        pq = np.zeros((num_samples, 2*num_bus))
        pq[:, np.arange(0, num_bus)] = p
        pq[:, np.arange(num_bus, 2*num_bus)] = q
        
        self.uw = np.copy(uw)
        self.pq = np.copy(pq)
        self.num_samples = num_samples
        self.num_bus = num_bus

    def train_test_split_data(self, rand_percent=True, time_series=False, train_percent=0.8):
        """
        Split the data into train and test sets according to two approaches.

        The first way of splitting the data, activated by rand_percent, selects train_percent percentage of the
        samples randomly to make up the training sets, X_train and y_train, and uses the remainder to form the testing
        sets, X_test and y_test. This is performed using the sklearn.model_selection.train_test_split.

        The second way of splitting the data, activated by time_series, simply takes the first train_percent chunk of
        the samples to make up the training sets and takes the following section for the testing sets. This is can
        be useful to preserve the time-series nature of the data and simulate how the training and testing process
        might play out in a real application.

        The data is named X and y following the sklearn convention for inputs and outputs, so our ML model will
        represent a function f: X -> y.

        Parameters
        ----------
        rand_percent: bool
            Whether to use the random way of splitting the data
        time_series: bool
            Whether to use the time series way of splitting the data
        train_percent: float
            What percentage of the data to include in the test set

        Attributes
        ----------
        X_train: array_like
            The input data for training
        X_test: array_like
            The input data for testing
        y_train: array_like
            The output data for training
        y_test: array_like
            The output data for testing
        num_train: int
            The size of the training set
        num_test: int
            The size of the testing set

        """
        if rand_percent:
            X_train, X_test, y_train, y_test = train_test_split(self.uw, self.pq, train_size=train_percent,
                                                                random_state=42)
            num_train = np.shape(X_train)[0]
            num_test = np.shape(X_test)[0]
        elif time_series:
            num_train = int(train_percent*np.shape(self.uw)[0])
            num_test = np.shape(self.uw)[0] - num_train
            X_train = self.uw[np.arange(0, num_train), :]
            X_test = self.uw[np.arange(num_train, num_train + num_test), :]
            y_train = self.pq[np.arange(0, num_train), :]
            y_test = self.pq[np.arange(num_train, num_train + num_test), :]
        else:
            print('Failed to split data properly. One of rand_percent and time_series must be True.')
            return

        self.X_train = np.copy(X_train)
        self.X_test = np.copy(X_test)
        self.y_train = np.copy(y_train)
        self.y_test = np.copy(y_test)
        self.num_train = num_train
        self.num_test = num_test
        
    def scale_data(self):
        
        self.X_means = np.mean(self.X_train, axis=0)
        self.X_stds = np.std(self.X_train, axis=0)
        self.y_means = np.mean(self.y_train, axis=0)
        self.y_stds = np.std(self.y_train, axis=0)
        
        for i in range(np.shape(self.X_train)[1]):
            if self.X_stds[i] != 0:
                self.X_train[:, i] = (self.X_train[:, i] - self.X_means[i])/self.X_stds[i]
                self.X_test[:, i] = (self.X_test[:, i] - self.X_means[i])/self.X_stds[i]
            else:
                self.X_train[:, i] = (self.X_train[:, i] - self.X_means[i])
                self.X_test[:, i] = (self.X_test[:, i] - self.X_means[i])
        for j in range(np.shape(self.y_train)[1]):
            if self.y_stds[j] != 0:
                self.y_train[:, j] = (self.y_train[:, j] - self.y_means[j])/self.y_stds[j]
                self.y_test[:, j] = (self.y_test[:, j] - self.y_means[j])/self.y_stds[j]
            else:
                self.y_train[:, j] = (self.y_train[:, j] - self.y_means[j])
                self.y_test[:, j] = (self.y_test[:, j] - self.y_means[j])
                
    def scale_back_sample_y(self, y_prediction, bus_number):
        
        true_y = y_prediction*self.y_stds[bus_number] + self.y_means[bus_number]

        return true_y

    def fit_svr(self, C_set, eps_set, max_iter, which_bus=None):
        """
        Train the SVR model to represent the power flow mapping f: X -> y at each bus and save the model parameters.

        This method builds a Support Vector Regression (SVR) model using sklearn.svm.SVR to predict the power
        injections at each bus in the network. sklearn.model_selection.GridSearchCV is used to find the optimal
        combination of C (regularization parameter) and epsilon (data significance measure) for each model, given the
        options in inputs C_set and eps_set. The data is preprocessed before fitting using
        sklearn.preprocessing.StandardScaler.

        The parameters of the trained models are collected into arrays and store as attributes of the model so that
        they can later be used for testing and prediction. Please see the documentation on sklearn.svm.SVR for more
        information on the different attributes of the trained model.

        Since the model is designed to estimate both the real and reactive power injections, p and q, there are
        2*num_bus separate models. The first num_bus models stored are for p and the second num_bus models are for q.

        Parameters
        ----------
        C_set, eps_set: list of float
            Values to try for C and epsilon, tunable parameters in the SVR model
        max_iter: int
            Maximum number of iterations to allow in the SVR fitting. This is used to limit the model training time.
        which_bus: list of int
            Optional input choosing which buses to build a model for.

        Attributes
        ----------
        b: array_like
            Intercept values for all the models, a num_models x 1 array where num_models = 2*num_bus
        coeffs: array_like
            Coefficients for all the kernel term for all the models, a num_models x num_train array
        num_SV: array_like
            The number of Support Vectors required for each of the models, a num_models x 1 array
        SV_inds: array_like
            The indices of which training samples were used as Support Vectors in each model, num_models x num_train
        C_best, eps_best: array_like
            The best C and epsilon values as chosen by GridSearchCV for each model, both num_models x 1 arrays

        """

        if which_bus is None: 
            which_bus = np.arange(0, 2*self.num_bus)
        
        b = np.zeros((2*self.num_bus, 1))
        coeffs = np.zeros((2*self.num_bus, self.num_train))
        num_SV = np.zeros((2*self.num_bus, 1))
        SV_inds = np.zeros((2*self.num_bus, self.num_train))
        C_best = np.zeros((2*self.num_bus, 1))
        eps_best = np.zeros((2*self.num_bus, 1))

        for l in range(np.shape(which_bus)[0]):# 2*self.num_bus):

            j = which_bus[l] # Bus index
            
            tuned_parameters = [{'kernel': ['poly'], 'degree': [2.0], 'C': C_set, 'epsilon': eps_set,
                                 'max_iter':[max_iter]}]

#             scaler_x = StandardScaler()
#             scaler_x.fit(self.X_train)
#             scaler_y = StandardScaler()
#             scaler_y.fit(self.y_train[:, j].reshape((self.num_train, 1)))

            regr_test = GridSearchCV(SVR(), tuned_parameters)
            regr_test.fit(self.X_train, self.y_train[:, j].reshape((self.num_train, )))
#             regr_test.fit(scaler_x.transform(self.X_train),
#                           (scaler_y.transform(self.y_train[:, j].reshape((self.num_train, 1))))
#                           .reshape((self.num_train,)))

            regr_svr = regr_test.best_estimator_
            eps_best[j] = regr_svr.epsilon
            C_best[j] = regr_svr.C
            b[j] = regr_svr.intercept_
            num_SV[j, 0] = np.size(regr_svr.support_)

            for i in np.arange(0, np.size(regr_svr.support_)):
                vals = regr_svr.dual_coef_
                coeffs[j, regr_svr.support_[i]] = vals.item(i)
                SV_inds[j, regr_svr.support_[i]] = 1
            
            self.best_svr_models[j] = regr_svr
                
        self.b = b
        self.coeffs = coeffs
        self.num_SV = num_SV
        self.SV_inds = SV_inds
        self.C_best = C_best
        self.eps_best = eps_best

    def apply_svr(self, X_sample, which_bus=None):
        """
        Apply the SVR model on the single input, X_sample, to estimate the power injections at each bus.

        The predicted value from each of the 2*num_bus models is calculated directly using the saved parameters b and
        coeffs, and sklearn.metrics.pairwise.polynomial_kernel is used to efficiently calculate the quadratic kernel
        keeping all the same parameters that were prescribed in the fitting.

        Applying the model to this un-scaled input requires regenerating the StandardScaler tool used in the training.
        scaler_x is generated using the original X_train used to fit the model and then is applied to the new X input,
        X_sample. scaler_y is generated using the original y_train used to fit the model and then is used to un-scale
        the predicted output, returning a prediction y_output that should approximate the true measurement data for
        that sample.

        Parameters
        ----------
        X_sample: array_like
            The input sample of cartesian voltage data on which we apply the model
        which_bus: list of int
            Optional input telling the method which buses there are models for

        Returns
        ----------
        y_output: array_like
            The output value of real and reactive power injections for this sample, estimated by the SVR model

        """
        
        if which_bus is None:
            which_bus = np.arange(0, 2*self.num_bus)

        y_output = np.zeros((2*self.num_bus,))
#         scaler_x = StandardScaler()
#         scaler_x.fit(self.X_train)        

        for i in range(np.shape(which_bus)[0]):  #2*self.num_bus):
            k = which_bus[i]
#             K = polynomial_kernel(self.X_train, X_sample.reshape((1, 2*self.num_bus)), degree=2, gamma=1.0, coef0=1.0)
#             K = polynomial_kernel(scaler_x.transform(self.X_train),
#                                   scaler_x.transform(X_sample.reshape((1, 2*self.num_bus))),
#                                   degree=2, gamma=1.0, coef0=1.0)
#             scaler_y = StandardScaler()
#             scaler_y.fit(self.y_train[:, k].reshape((self.num_train, 1)))
#             y_output[k] = np.reshape(np.mat(self.coeffs[k, :])*np.mat(K) + self.b[k, 0], (1, 1))
#             y_output[k] = scaler_y.inverse_transform((np.mat(self.coeffs[k, :])*np.mat(K) + self.b[k, 0]).reshape((1, 1)))
            y_output[k] = self.best_svr_models[k].predict(X_sample.reshape((1, 2*self.num_bus)))

        return y_output

    def test_error_svr(self, which_bus=None):
        """
        Calculate the test error of the model over the full test set.

        This method runs through all the num_test samples in the test set, applies the fitted model to each sample, and
        compares the models estimates of the power injections against the true values from y_test. A measure of the
        total root mean squared error, total_rmse, is used to quantify the difference between the estimates and
        measured values over the whole set.
        
        Parameters
        ----------
        which_bus: list of int
            Optional input telling the method which bus models to test.

        Attributes
        ----------
        test_y_values: array_like
            The output values estimated by the model for each sample
        test_error_values: array_like
            The RMSE error for each sample between the estimated and measured values
        total_rmse: float
            The total RMSE error in the estimates from the test set

        """

        test_error_values = np.zeros((self.num_test, 1))
        test_y_values = np.zeros((self.num_test, 2 * self.num_bus))

        for i in range(self.num_test):

            test_y_values[i, :] = self.apply_svr(self.X_test[i, :], which_bus)
            test_error_values[i] = np.linalg.norm(test_y_values[i, :] -
                                                  self.y_test[i, :], 2) / np.power(2 * self.num_bus, 0.5)

        total_rmse = np.sqrt(np.mean(np.power(test_y_values - self.y_test, 2)))
        
        self.test_y_values = test_y_values
        self.test_error_values = test_error_values
        self.total_rmse = total_rmse


