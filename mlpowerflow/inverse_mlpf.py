import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')


class InverseMLPF(object):

    """
    Build, train, implement, and test an ML model of the * inverse * power flow equations, also known as voltage
    estimation.

    Please refer to the README.md for more information about the researchers and theory.

    The power flow functions define the map between voltages and power injections. In this implementation, two models
    are used to replace the power flow equations with data-driven models of the mapping: a Support Vector Regression
    (SVR) model with a quadratic kernel, or a simple Linear Regression (LR). All of the models parameters are trained
    directly from data taken at all of the buses in the network, without any direct knowledge of the line parameters
    or Ybus matrix.

    The methods in this class ingest and process voltage and power injection data, fit a model for the voltage
    estimation mappings at each bus in the network, and test the mapping using a reserved test set.

    """

    def __init__(self, num_bus, num_samples):
        """Initialize attributes of the object."""

        self.num_bus = num_bus
        self.num_samples = num_samples
        self.p = np.zeros((1, 1))
        self.q = np.zeros((1, 1))
        self.v = np.zeros((1, 1))
        self.a = np.zeros((1, 1))
        self.pq = np.zeros((1, 1))
        self.num_train = 0
        self.num_test = 0
        self.X_train = np.zeros((1, 1))
        self.y_train = np.zeros((1, 1))
        self.X_test = np.zeros((1, 1))
        self.y_test = np.zeros((1, 1))
        self.test_y_values_svr = np.zeros((1, 1))
        self.test_error_values_svr = np.zeros((1, 1))
        self.total_rmse_svr = np.zeros((1, 1))
        self.test_y_values_lr = np.zeros((1, 1))
        self.test_error_values_lr = np.zeros((1, 1))
        self.total_rmse_lr = np.zeros((1, 1))
        self.best_svr_models = {}
        self.b = np.zeros((1, 1))
        self.coeffs = np.zeros((1, 1))
        self.num_SV = np.zeros((1, 1))
        self.SV_inds = np.zeros((1, 1))
        self.C_best = np.zeros((1, 1))
        self.eps_best = np.zeros((1, 1))
        self.lr_coeffs = np.zeros((1, 1))
        self.lr_intercept = np.zeros((1, 1))

    def supply_full_data(self, p, q, v, num_bus, num_samples, a=None):
        """
        Ingest the p, q, v, and a data into the object. Manipulate it into the input - output formats.

        The power variables p and q are combined and saved together in pq to be used as the input
        of the voltage estimation model. It is anticipated that voltage phase angle data might be unavailable so in that
        case the phase angle values default to zero.

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
        pq: array_like
            Combined power injections
        """

        self.p = np.copy(p)
        self.q = np.copy(q)
        self.v = np.copy(v)
        if a is None:
            self.a = np.zeros(np.shape(v))
        else:
            self.a = np.deg2rad(np.copy(a))

        # Joined p and q for output
        pq = np.zeros((num_samples, 2 * num_bus))
        pq[:, np.arange(0, num_bus)] = p
        pq[:, np.arange(num_bus, 2 * num_bus)] = q

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

        Note that for this case, the inverse mapping, X now represents the real and reactive power injections store in
        self.pq and y represents the voltage magnitudes stored in self.v.

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
            X_train, X_test, y_train, y_test = train_test_split(self.pq, self.v, train_size=train_percent,
                                                                random_state=42)
            num_train = np.shape(X_train)[0]
            num_test = np.shape(X_test)[0]
        elif time_series:
            num_train = int(train_percent*np.shape(self.pq)[0])
            num_test = np.shape(self.pq)[0] - num_train
            X_train = self.pq[np.arange(0, num_train), :]
            X_test = self.pq[np.arange(num_train, num_train + num_test), :]
            y_train = self.v[np.arange(0, num_train), :]
            y_test = self.v[np.arange(num_train, num_train + num_test), :]
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
        Train the SVR model to represent the inverse power flow mapping f: X -> y at each bus and save the
        model parameters.

        This method builds a Support Vector Regression (SVR) model using sklearn.svm.SVR to predict the voltage
        magnitude at each bus in the network. sklearn.model_selection.GridSearchCV is used to find the optimal
        combination of C (regularization parameter) and epsilon (data significance measure) for each model, given the
        options in inputs C_set and eps_set. The data is preprocessed before fitting using
        sklearn.preprocessing.StandardScaler.

        The parameters of the trained models are collected into arrays and store as attributes of the model so that
        they can later be used for testing and prediction. Please see the documentation on sklearn.svm.SVR for more
        information on the different attributes of the trained model.

        Since the model is designed to estimate just the voltage magnitude, there are num_bus separate models. The size
        of the input vectors is then 2*num_bus since both the real and reactive power injections, p and q, are used.
        Simple extensions of this work could incorporate the voltage phase angle, either as an input or output.

        Parameters
        ----------
        C_set, eps_set: list of float
            Values to try for C and epsilon, tunable parameters in the SVR model
        max_iter: int
            Maximum number of iterations to allow in the SVR fitting. This is used to limit the model training time.

        Attributes
        ----------
        b: array_like
            Intercept values for all the models, a num_models x 1 array where num_models = num_bus (different from
            ForwardMLPF case where predicted 2*num_bus values)
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

        for l in range(np.shape(which_bus)[0]): #self.num_bus):
            j = which_bus[l]

            tuned_parameters = [{'kernel': ['poly'], 'degree': [2.0], 'C': C_set, 'epsilon': eps_set,
                                 'max_iter': [max_iter]}]

            regr_test = GridSearchCV(SVR(), tuned_parameters)
            regr_test.fit(self.X_train, self.y_train[:, j].reshape((self.num_train,)))

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
            print('Done training voltage model for bus ', j, ', with a total of ',np.size(regr_svr.support_),
                  ' support vectors.')

        self.b = b
        self.coeffs = coeffs
        self.num_SV = num_SV
        self.SV_inds = SV_inds
        self.C_best = C_best
        self.eps_best = eps_best

    def fit_lr(self):
        """
        Train a Linear Regression (LR) model to represent the inverse power flow mapping f: X -> y at each bus and
        save the model parameters.

        This method builds an LR model using sklearn.linear_model.LinearRegression to estimate the voltage magnitude
        at each bus in the network. The data is preprocessed during the fitting, so no StandardScaler is used.

        The parameters of the trained models are collected into arrays and store as attributes of the model so that
        they can later be used for testing and prediction. Please see the documentation on
        sklearn.linear_model.LinearRegression for more information on the different attributes of the trained model.

        Since the model is designed to estimate just the voltage magnitude, there are num_bus separate models. The size
        of the input vectors is then 2*num_bus since both the real and reactive power injections, p and q, are used.
        Simple extensions of this work could incorporate the voltage phase angle, either as an input or output.

        Attributes
        ----------
        b: array_like
            Intercept values for all the models, a num_models x 1 array where num_models = num_bus (different from
            ForwardMLPF case where predicted 2*num_bus values)
        coeffs: array_like
            Coefficients for all the kernel term for all the models, a num_models x num_train array
        num_SV: array_like
            The number of Support Vectors required for each of the models, a num_models x 1 array
        SV_inds: array_like
            The indices of which training samples were used as Support Vectors in each model, num_models x num_train
        C_best, eps_best: array_like
            The best C and epsilon values as chosen by GridSearchCV for each model, both num_models x 1 arrays

        """

        lr_coeffs = np.zeros((self.num_bus, 2 * self.num_bus))
        lr_intercept = np.zeros((self.num_bus, 1))

        for j in range(self.num_bus):

            regr_lr = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
            regr_lr.fit(self.X_train, self.y_train[:, j])

            lr_coeffs[j, :] = regr_lr.coef_
            lr_intercept[j, 0] = regr_lr.intercept_

        self.lr_coeffs = lr_coeffs
        self.lr_intercept = lr_intercept

    def apply_svr(self, X_sample, which_buses=None):
        """
        Apply the SVR model on the single input, X_sample, to estimate the voltage magnitude at each bus.

        The predicted value from each of the num_bus models is calculated directly using the saved parameters b and
        coeffs, and sklearn.metrics.pairwise.polynomial_kernel is used to efficiently calculate the quadratic kernel
        keeping all the same parameters that were prescribed in the fitting.

        Applying the model to this un-scaled input requires regenerating the StandardScaler tool used in the training.
        scaler_x is generated using the original X_train used to fit the model and then is applied to the new X input,
        X_sample. scaler_y is generated using the original y_train used to fit the model and then is used to un-scale
        the predicted output, returning a prediction y_output that should approximate the true measurement data for
        that sample.

        The optional input makes it possible to estimate the voltage at a subset of the buses. This would be useful, for
        example, if one wanted to only predict values for the sub-station aggregation point.

        Parameters
        ----------
        X_sample: array_like
            The input sample of cartesian voltage data on which we apply the model
        which_buses: array_like
            The numbers of the buses for which we want to apply the model and estimate the voltage

        Returns
        ----------
        y_output: array_like
            The output value of voltage magnitude for this sample, estimated by the SVR model

        """

        if which_buses is None:
            which_buses = range(2 * self.num_bus)

        y_output = np.zeros((np.shape(which_buses)[0],))

        for j in range(np.shape(which_buses)[0]):
            k = which_buses[j]
            y_output[j] = self.best_svr_models[k].predict(X_sample.reshape((1, 2*self.num_bus)))

        return y_output

    def apply_lr(self, X_sample, which_buses=None):
        """
        Apply the LR model on the single input, X_sample, to estimate the voltage magnitude at each bus.

        The predicted value from each of the num_bus models is calculated directly using the saved parameters.

        The optional input makes it possible to estimate the voltage at a subset of the buses. This would be useful, for
        example, if one wanted to only predict values for the sub-station aggregation point.

        Parameters
        ----------
        X_sample: array_like
            The input sample of cartesian voltage data on which we apply the model
        which_buses: array_like
            The numbers of the buses for which we want to apply the model and estimate the voltage

        Returns
        ----------
        y_output: array_like
            The output value of voltage magnitude for this sample, estimated by the LR model

        """

        if which_buses is None:
            which_buses = range(2 * self.num_bus)

        y_output = (self.lr_coeffs.dot(X_sample.reshape(np.shape(self.lr_coeffs)[1], 1)) + self.lr_intercept).ravel()
        y_output = y_output[which_buses]

        return y_output

    def test_error_svr(self, which_buses=None):
        """
        Calculate the test error of the model over the full test set for all or a subset of the buses.

        This method runs through all the num_test samples in the test set, applies the fitted model to each sample, and
        compares the models estimates of the voltage magnitude against the true values from y_test. A measure of the
        total root mean squared error, total_rmse, is used to quantify the difference between the estimates and
        measured values over the whole set.

        Attributes
        ----------
        test_y_values: array_like
            The output values estimated by the model for each sample
        test_error_values: array_like
            The RMSE error for each sample between the estimated and measured values
        total_rmse: float
            The total RMSE error in the estimates from the test set

        """

        if which_buses is None:
            which_buses = range(2 * self.num_bus)

        test_error_values = np.zeros((self.num_test, 1))
        test_y_values = np.zeros((self.num_test, np.shape(which_buses)[0]))

        for i in range(self.num_test):
            test_y_values[i, :] = self.apply_svr(self.X_test[i, :], which_buses)
            test_error_values[i] = np.linalg.norm(test_y_values[i, :] -
                                                  self.y_test[i, :], 2) / np.power(np.shape(which_buses)[0], 0.5)

        total_rmse = np.sqrt(np.mean(np.power(test_y_values - self.y_test, 2)))

        self.test_y_values_svr = test_y_values
        self.test_error_values_svr = test_error_values
        self.total_rmse_svr = total_rmse

    def test_error_lr(self, which_buses=None):
        """
        Calculate the test error of the model over the full test set for all or a subset of the buses with the LR model.

        This method runs through all the num_test samples in the test set, applies the fitted model to each sample, and
        compares the models estimates of the voltage magnitude against the true values from y_test. A measure of the
        total root mean squared error, total_rmse, is used to quantify the difference between the estimates and
        measured values over the whole set.

        Attributes
        ----------
        test_y_values: array_like
            The output values estimated by the model for each sample
        test_error_values: array_like
            The RMSE error for each sample between the estimated and measured values
        total_rmse: float
            The total RMSE error in the estimates from the test set

        """

        if which_buses is None:
            which_buses = range(2 * self.num_bus)

        test_error_values = np.zeros((self.num_test, 1))
        test_y_values = np.zeros((self.num_test, np.shape(which_buses)[0]))  # 2 * self.num_bus))

        for i in range(self.num_test):
            test_y_values[i, :] = self.apply_lr(self.X_test[i, :], which_buses)
            test_error_values[i] = np.linalg.norm(test_y_values[i, :] -
                                                  self.y_test[i, :], 2) / np.power(np.shape(which_buses)[0], 0.5)

        total_rmse = np.sqrt(np.mean(np.power(test_y_values - self.y_test, 2)))

        self.test_y_values_lr = test_y_values
        self.test_error_values_lr = test_error_values
        self.total_rmse_lr = total_rmse
