import numpy as np
import pandas as pd
import sys, os
from os.path import join
from multiprocessing import Process, Lock, Event
import multiprocessing
import time
import psutil
import pickle
from skimage.util.shape import view_as_windows
import datetime
# ===============================================
# methods
# import ErcotLikeDays as ercot
import GradientBoosting as gbr
import LassoLars as lasso
import AdaptiveLinear as al

# ===============================================

np.set_printoptions(precision=3)


class runModels:
    """
    A wrapper to evaluate different forecasting methods
    """

    def __init__(self, datetime, path, typeModel, windowSize):
        """
        Initialization function
        :param datetime:
        :param path:
        :param typeModel:
        :param windowSize:
        """
        self.dates = pd.DatetimeIndex(datetime.to_pydatetime())
        self.path = path
        self.typeModel = typeModel
        self.windowSize = windowSize

    def joinAllResults(self, path, nUsers, stepAhead):
        user = 0
        key = str(user) + '_' + str(stepAhead) + 'h'
        df_train_main = pd.read_csv((path + str(stepAhead) + 'h_user' + str(user) + '_train.csv'), index_col=0)
        df_test_main = pd.read_csv((path + str(stepAhead) + 'h_user' + str(user) + '_test.csv'), index_col=0)

        for user in xrange(1, nUsers):
            key = str(user) + '_' + str(stepAhead) + 'h'

            df_test = pd.read_csv((path + str(stepAhead) + 'h_user' + str(user) + '_test.csv'), index_col=0)
            df_test_main[key] = df_test[key]
            os.remove((path + str(stepAhead) + 'h_user' + str(user) + '_test.csv'))

            df_train = pd.read_csv((path + str(stepAhead) + 'h_user' + str(user) + '_train.csv'), index_col=0)
            df_train_main[key] = df_train[key]
            os.remove((path + str(stepAhead) + 'h_user' + str(user) + '_train.csv'))
        df_test_main.to_csv(self.path + '/prediction_' + str((stepAhead)) + 'h_test.csv', float_format='%.3f')
        df_train_main.to_csv(self.path + '/prediction_' + str((stepAhead)) + 'h_train.csv', float_format='%.3f')
        os.remove((path + str(stepAhead) + 'h_user0_test.csv'))
        os.remove((path + str(stepAhead) + 'h_user0_train.csv'))

    def createExportFileUser(self, nHour, nTrain, user, hour, y_forecast_te, y_forecast_tr, path):
        key = str(user) + '_' + str(hour) + 'h'
        index = nTrain + nHour + (hour - 1)
        df_ = pd.DataFrame(index=self.dates[index:], columns=[key])
        df_[key] = y_forecast_te
        df_.to_csv(path + '_test.csv', float_format='%.3f')

        index1 = nHour + (hour - 1)
        df_ = pd.DataFrame(index=self.dates[index1:index], columns=[key])
        df_[key] = y_forecast_tr
        df_.to_csv(path + '_train.csv', float_format='%.3f')

    def exportTrainTest(self, dates, nTrain, y_actual):
        listCol = []
        y_actual_tr = y_actual[:nTrain, :]
        y_actual_te = y_actual[nTrain:, :]
        nUsers = y_actual_te.shape[1]
        for user in xrange(nUsers):
            listCol.append(user)
        df_train = pd.DataFrame(index=dates[:nTrain], columns=listCol)
        df_test = pd.DataFrame(index=dates[nTrain:], columns=listCol)
        for user in xrange(nUsers):
            df_train[user] = y_actual_tr[:, user]
            df_test[user] = y_actual_te[:, user]
        df_train.to_csv(self.path + '/real_train.csv', float_format='%.3f')
        df_test.to_csv(self.path + '/real_test.csv', float_format='%.3f')

    def getRMSE(self, y, y_predict):
        return np.sqrt((np.power(y_predict - y, 2)).sum() / y_predict.size)

    def getMSE(self, y, y_predict):
        # Mean squared error
        return ((np.power(y_predict - y, 2)).sum() / (y_predict.size * 1.0))

    def runHourUser(self, user, load, nTrain, stepAhead, dates, xTemp, path):
        """
        Load forecasting for one user
        :param user:
        :param load: load for one user
        :param nTrain:
        :param stepAhead:
        :param dates:
        :param xTemp:
        :param path:
        :return:
        """
        # The path for storing testing loads
        pathTest = path + str((stepAhead)) + 'h_user' + str(user) + '_test.csv'

        if not os.path.exists(pathTest):

            window_shape = (self.windowSize,)
            xLoad = (view_as_windows(load.copy(), window_shape))[:-stepAhead, :]
            yLoad = load[(self.windowSize + stepAhead - 1):]
            y = yLoad.ravel()
            # 168 Temp | 168 Load
            X = np.empty((xLoad.shape[0], (2 * self.windowSize)), dtype='float')
            X[:, :self.windowSize] = xTemp  # 168 lagged temp
            X[:, self.windowSize:(2 * self.windowSize)] = xLoad  # 168 lagged load
            model = None
            if self.typeModel == 'LassoLarsIC':
                model = lasso.LassoLarsIC()
            elif self.typeModel == 'GradientBoostingRegressor':
                model = gbr.GradientBoosting()
            elif self.typeModel == 'AdaptiveLinear':
                model = al.AdaptiveLinear(y)
            # elif self.typeModel == 'ErcotLikeDays':
            #    model = ercot.ErcotLikeDays()
            X = model.generate_x(X, dates, stepAhead)
            model.fit(X[:nTrain, :], y[:nTrain])
            if self.typeModel == 'AdaptiveLinear':
                y_train = model.predict(X[:nTrain, :], y[:nTrain])
                y_test = model.predict(X[nTrain:, :], y[nTrain:])
            else:
                y_train = model.predict(X[:nTrain, :])
                y_test = model.predict(X[nTrain:, :])

            # y_test = np.empty_like(y[nTrain:])
            # for t in xrange(X[nTrain:,:].shape[0]):
            #    y_test[t] = model.predict(np.expand_dims(X[nTrain+t,:],axis=0))
            self.createExportFileUser(self.windowSize, nTrain, user, stepAhead, y_test, y_train,
                                      path + str((stepAhead)) + 'h_user' + str(user))
        else:
            print 'user already calculated.'
        print 'finished user', user

    def checkAllPred(self, nUsers, setT, path):
        for user in xrange(nUsers):
            if not os.path.exists(path + str(user) + '_' + setT + '.csv'):
                print 'missing user=' + str(user) + ' ' + setT
                return True
        return False

    def runPredictionHour(self, load, user2zip, stepAhead, xTemp, sizeLastMonth):
        """
        A wrapper for load forecasting for multi users
        :param load: a numpy array, size(length of all training and testing, number of users)
        :param user2zip: a numpy array, size (number of users, 1), the zip5 of each user
        :param stepAhead: an int, forecasting horizon
        :param xTemp: a dict, keys are zips (int), values are associated numpy array,
                      size(window size, length of training and testing)
        :param sizeLastMonth: an int, the total data points for last hour
        :return: None
        """
        path_prefix = self.path + '/prediction_'

        # Date time range for training and testing
        dates = self.dates[self.windowSize + (stepAhead - 1):]
        nUsers = load.shape[1]

        N = xTemp[user2zip[0]].shape[0]  # xTemp.shape[0], N == dates.size, number of training and testing samples

        nTrain = N - sizeLastMonth # Number of training samples

        if not (os.path.isfile(self.path + '/real_test.csv')) or not (os.path.isfile(self.path + '/real_train.csv')):
            print 'exporting train and test'
            self.exportTrainTest(self.dates, nTrain + self.windowSize, load)  # ok
        print '-------------------------------------------------------'
        print 'starting parallel processing.'
        print 'number of CPUs = ', multiprocessing.cpu_count()
        print '-------------------------------------------------------'

        '''
        user=23
        self.runHourUser(user, load, nHour, nTrain, stepAhead, dates, xTemp[user2zip[user]], path)
        aa/bb
        '''
        numProcess = 30
        for user in xrange(nUsers):
            print 'user=', user
            Process(target=self.runHourUser,
                    args=(user, load[:, user], nTrain, stepAhead, dates, xTemp[user2zip[user]], path_prefix)).start()
            while (len(multiprocessing.active_children()) > numProcess):
                time.sleep(1)
            if psutil.virtual_memory()[2] > 80.0:
                print 'waiting because of memory...'
                print 'memory = ', psutil.virtual_memory()[2]
                while (psutil.virtual_memory()[2]) > 50.0:
                    time.sleep(5)
                print 'restarting.'
        pathCheck = path_prefix + str((stepAhead)) + 'h_user'
        while (self.checkAllPred(nUsers, 'test', pathCheck) and self.checkAllPred(nUsers, 'train', pathCheck)):
            time.sleep(2)

        print 'joinAllResults...'
        self.joinAllResults(path_prefix, nUsers, stepAhead)
        print 'finished hour ', stepAhead


if __name__ == '__main__':

    data_dir = '/home/mateus/S3L/1year_aligned'
    working_dir = join(data_dir, 'set_120k')

    hours = np.arange(1, 25)  # each hour to predict

    typeModels = np.array(['AdaptiveLinear'])
    # np.array(['GradientBoostingRegressor','LassoLarsIC'])#str(sys.argv[1]) # all differents models that we want to test

    dates = pd.date_range('08/01/2011', periods=8760, freq='H')  # this is the date-time of data.

    tempPath = join(data_dir + 'temp_by_zip.csv')
    df_tempByZip = pd.read_csv(tempPath, index_col=0)
    windowSize = 168
    window_shape = (windowSize,)
    discretization = 24
    pathZip = join(data_dir + 'zip5.csv')
    loadFile = np.loadtxt(pathZip, delimiter=',', dtype=np.str)
    user_to_zip = loadFile[1:, 1].astype('int')
    num_users_per_grp = 2000

    # getting the number of days for each month
    month_end_dates = (dates[dates.is_month_end])[::24]
    numberDaysMonth = np.array(month_end_dates.day)
    # For the case that the last day of the data is not the last day of the month,
    # for example: if data has 8760 time stamps and it is leap year, then last day is 7/30
    if month_end_dates[-1].month != dates[-1].month:
        numberDaysMonth = np.concatenate([numberDaysMonth, (np.array([dates[-1].day]))])

    sizeFrameMonth = 3  # 3 months train and one extra for test

    # Generate 2D arrays representing each training months (Last one does not have test data)
    xMonths = view_as_windows(np.arange(numberDaysMonth.size), (sizeFrameMonth,))

    # Iterate 59 groups
    for group_idx in xrange(1, 60):
        print 'Starting group ', group_idx
        # Read load file for one group
        loadPath = data_dir + '/user_grp_' + str(group_idx) + '.csv'
        loadFile = np.loadtxt(loadPath, delimiter=',', dtype=np.str)
        load = loadFile[1:, 1:].astype('float')
        # Remove empty data
        idx = np.logical_or(np.isnan(load), np.isinf(load))
        load[idx] = 0.0

        # Get zip for each user in the group
        zipcodes = user_to_zip[num_users_per_grp * (group_idx - 1):num_users_per_grp * (group_idx)]

        # Iterater over each subset 3 months train and 1 test.
        for training_set_idx in xrange(1):
        # for training_set_idx in xMonths.shape[0]-1): # Last tuple does not have testing data available.

            # Get the start index and end index, from 0 to 8760
            sub_start = numberDaysMonth[:training_set_idx].sum() * discretization
            # End index is sizeFrameMonth + 1 month to sub_start
            sub_end = numberDaysMonth[:(training_set_idx + sizeFrameMonth + 1)].sum() * discretization

            sizeLastMonth = numberDaysMonth[(training_set_idx + sizeFrameMonth)] * discretization

            print 'calculating xTemp for all zipcodes.'
            xTemp_by_hour = {}
            for h in hours:
                xTemp_by_hour[h] = {}
            for zipcode in df_tempByZip.keys():
                serie = np.array(df_tempByZip[zipcode][sub_start:sub_end])
                matrixTemp = view_as_windows(serie, window_shape)

                for h in hours:
                    xTemp_by_hour[h][int(zipcode)] = matrixTemp[:-h, :]
            print 'finished all xTemp'

            for typeModel in typeModels:

                pathForecast = join(working_dir,
                                    typeModel,
                                    str(dates[sub_end - 1].month) + '_' + str(dates[sub_end - 1].year),
                                    'user_grp_' + str(i))

                if not os.path.exists(pathForecast):
                    os.makedirs(pathForecast)

                # Initialize a class instance
                # Given: user_grp, 3 month subset, Model
                forecast = runModels(dates[sub_start:sub_end], pathForecast, typeModel, windowSize)

                for hour in hours:
                    fileForecast = 'prediction_' + str(hour) + 'h_test.csv'
                    if not os.path.exists(join(pathForecast, fileForecast)):
                        forecast.runPredictionHour(load[sub_start:sub_end], zipcodes, hour, xTemp_by_hour[hour],
                                                   sizeLastMonth)

                    print 'Finish hour ', hour
                print 'finished subsample ', training_set_idx
        print 'Finished group ', group_idx

    print 'finished script'
