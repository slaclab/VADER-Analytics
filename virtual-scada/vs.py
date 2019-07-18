import numpy as np
import pandas as pd
import sys

sys.path.insert(0, 'C:/Users/Serhan/Documents/SLACwork/VADER-Analytics/mlpowerflow')

import forward_mlpf
import inverse_mlpf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def removeValues(data, percentage, inplace = False):
    """
    Removes individual values in a pandas dataframe with probabilty percentage

    Inputs:
        data: pandas dataframe to remove values from
        percentage: probability with which to remove an individual value
        inplace: whether to modify the data in place or not, false by default

    Output:
        dataRemoved: the corresponding data with the corresponding values removed

    """

    if inplace:
        dataRemoved = data
    else:
        dataRemoved = data.copy()

    numRows, numCols = data.shape
    for i in range(numCols):
        for j in range(numRows):
            if np.random.uniform() < percentage:
                dataRemoved[i][j] = None


    return dataRemoved

def removeRows(data, rowPercentage, colPercentage = 1, inplace = False):
    """
    Selects rows with probability rowPercentage, then for each column removes that column
    from that row with probability ColPercentage

    Inputs:
        data: pandas dataframe to remove values from
        rowPercentage: probability with which to select a row to remove data from
        colPercentage: probability with which to select a column to remove data from, 1 by default
        inplace: whether to modify the data in place or not, false by default

    Output:
        dataRemoved: the corresponding data with the corresponding values removed
    """

    if inplace:
        dataRemoved = data
    else:
        dataRemoved = data.copy()

    numRows, numCols = data.shape
    for j in range(numRows):
        if np.random.uniform() < rowPercentage:
            for i in range(numCols):
                if np.random.uniform() < colPercentage:
                    dataRemoved[i][j] = None

    return dataRemoved


def nonNullIntersection(dataList):
    """
    Takes a list of dataframes and returns the list of dataframes with rows that contain a null value in
    any dataframe removed. This is so that the row can be used for training ML Powerflow

    Input:
        dataList: a list of pandas dataframes of the same number of rows
    Output:
        dataListIntersection: a list of pandas dataframes with the rows that contain a null values
        in any row missing.
    """

    masks = []
    for data in dataList:
        mask = data.isna().any(axis=1)
        masks.append(~mask)

    finalMask = masks[0]
    for mask in masks[1:]:
        finalMask = finalMask & mask

    dataListIntersection = []

    for x in range(len(dataList)):
        dataListIntersection.append(dataList[x][finalMask])

    return dataListIntersection

def fillValuesMLPFForward(p, q, v, a, max_iter = 1e3, C_set = [1], eps_set = [1e-3]):
    """
    Takes real and reactive power and voltage and phase angle and trains
    an mlpowerflow model to fill in missing power data.

    Does this by first filtering out all rows that can't be trained on.
    Then train the mlpowerflow model on that data.
    Then fill in the missing data

    Input:
        p: the real power matrix
        q: the reactive power matrix
        v: the voltage matrix
        a: the phase angle matrix
        max_iter: the maximum number of steps to run for training the model
        C_set: the set of C values to try while training the model
        eps_set: the set of epsilon values to try while training the model
    Ouput:
        pFilled: the real power matrix with missing values filled in
        qFilled: the reactive power matrix with missing values filled in
    """
    import warnings
    warnings.filterwarnings("ignore")

    # First, we need a full row of voltage and power to train our mlpowerflow model
    dataNonNull = nonNullIntersection([p,q,v,a])
    pNonNull = dataNonNull[0].values
    qNonNull = dataNonNull[1].values
    vNonNull = dataNonNull[2].values
    aNonNull = dataNonNull[3].values

    # Now we train the model
    num_samples, num_bus = pNonNull.shape
    model = forward_mlpf.ForwardMLPF(num_bus, num_samples)

    model.supply_full_data(pNonNull, qNonNull, vNonNull, aNonNull, num_bus, num_samples)

    model.train_test_split_data(train_percent = .9)
    model.scale_data()

    model.fit_svr(C_set, eps_set, max_iter)
    model.test_error_svr()
    #print(model.scaled_total_rmse, model.mean_rmse)

    # Finally we fill all the missing data
    voltage = pd.concat([v,a],axis=1).values
    power = pd.concat([p,q],axis=1).values

    scaler_x = StandardScaler()
    scaler_x.fit(model.X_train)

    for j in range(num_samples):
        if np.isnan(np.sum(power[j]))and not np.isnan(np.sum(voltage[j])):
            vj = voltage[j][0:30]
            aj = voltage[j][30:]
            u = vj * np.cos(aj)  # Make sure a is in radians
            w = vj * np.sin(aj)
            uw = np.zeros((1, 2 * num_bus))
            uw[:, np.arange(0, num_bus)] = u
            uw[:, np.arange(num_bus, 2 * num_bus)] = w

            XScaled = scaler_x.transform(uw)

            predictions = model.apply_svr(XScaled)


            for i in range(2*num_bus):
                if np.isnan(power[j,i]):
                    pred = predictions[i]
                    scaled = model.scale_back_sample_y(pred, i)

                    power[j,i] = scaled

    pFilled = pd.DataFrame(power[:, :num_bus])
    qFilled = pd.DataFrame(power[:, num_bus:])

    return pFilled, qFilled

def fillValuesMLPFInverse(p, q, v, a, max_iter = 1e3, C_set = [1], eps_set = [1e-3]):
    """
        Takes real and reactive power and voltage and phase angle and trains
        an mlpowerflow model to fill in missing power data.

        Does this by first filtering out all rows that can't be trained on.
        Then train the mlpowerflow model on that data.
        Then fill in the missing data

    Input:
        p: the real power matrix
        q: the reactive power matrix
        v: the voltage matrix
        a: the phase angle matrix
        max_iter: the maximum number of steps to run for training the model
        C_set: the set of C values to try while training the model
        eps_set: the set of epsilon values to try while training the model
    Ouput:
        vFilled: the voltage matrix with missing values filled in
        aFilled: the phasne angle matrix with missing values filled in
    """
    import warnings
    warnings.filterwarnings("ignore")

    dataNonNull = nonNullIntersection([p, q, v, a])
    pNonNull = dataNonNull[0].values
    qNonNull = dataNonNull[1].values
    vNonNull = dataNonNull[2].values
    aNonNull = dataNonNull[3].values

    num_samples, num_bus = pNonNull.shape
    model = inverse_mlpf.InverseMLPF(num_bus, num_samples)

    model.supply_full_data(pNonNull, qNonNull, vNonNull, num_bus, num_samples, a = aNonNull)

    model.train_test_split_data(train_percent=.9)
    model.scale_data()
    model.fit_svr(C_set, eps_set, max_iter)
    model.test_error_svr()
    #print(model.scaled_total_rmse, model.mean_rmse)

    power = pd.concat([p, q], axis=1).values
    voltage = pd.concat([v, a], axis=1).values

    scaler_x = StandardScaler()
    scaler_x.fit(model.X_train)

    for j in range(num_samples):
        if np.isnan(np.sum(voltage[j])) and not np.isnan(np.sum(power[j])):
            XScaled = scaler_x.transform(power[j].reshape(1,-1))
            predictions = model.apply_svr(XScaled)

            for i in range(num_bus):
                if np.isnan(voltage[j, i]):

                    scaled = model.scale_back_sample_y(predictions[i], i)
                    voltage[j, i] = scaled



    vFilled = pd.DataFrame(voltage[:, :num_bus])
    aFilled = pd.DataFrame(voltage[:, num_bus:])

    return vFilled, aFilled

def fillValuesLRForward(p, q, v, a):
    """
        Takes voltage and phase angle and trains
        a linear regression model to fill in missing power data.

        Does this by first filtering out all rows that can't be trained on.
        Then train the model on that data.
        Then fill in the missing data

    Input:
        p: the real power matrix
        q: the reactive power matrix
        v: the voltage matrix
        a: the phase angle matrix
    Ouput:
        pFilled: the real power matrix with missing values filled in
        qFilled: the reactive power matrix with missing values filled in
    """


    dataNonNull = nonNullIntersection([p, q, v, a])
    pNonNull = dataNonNull[0].values
    qNonNull = dataNonNull[1].values
    vNonNull = dataNonNull[2].values
    aNonNull = dataNonNull[3].values

    num_samples, num_bus = pNonNull.shape

    pq = np.zeros((num_samples, 2 * num_bus))
    pq[:, np.arange(0, num_bus)] = pNonNull
    pq[:, np.arange(num_bus, 2 * num_bus)] = qNonNull

    u = vNonNull * np.cos(aNonNull)
    w = vNonNull * np.sin(aNonNull)
    uw_train = np.zeros((num_samples, 2 * num_bus))
    uw_train[:, np.arange(0, num_bus)] = u
    uw_train[:, np.arange(num_bus, 2 * num_bus)] = w

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaledX = scaler_x.fit_transform(uw_train)
    scaledY = scaler_y.fit_transform(pq)

    model = LinearRegression().fit(scaledX, scaledY)

    power = pd.concat([p, q], axis=1).values
    voltage = pd.concat([v, a], axis=1).values

    for j in range(num_samples):
        if np.isnan(np.sum(power[j])) and not np.isnan(np.sum(voltage[j])):
            vj = voltage[j][0:30]
            aj = voltage[j][30:]
            u = vj * np.cos(aj)  # Make sure a is in radians
            w = vj * np.sin(aj)
            uw = np.zeros((1, 2 * num_bus))
            uw[:, np.arange(0, num_bus)] = u
            uw[:, np.arange(num_bus, 2 * num_bus)] = w

            uw_scaled = scaler_x.transform(uw)

            predictions = model.predict(uw_scaled)

            for i in range(2 * num_bus):
                if np.isnan(power[j, i]):
                    pred = predictions[0]
                    scaled = scaler_y.inverse_transform(pred)[i]

                    power[j, i] = scaled

    pFilled = pd.DataFrame(power[:, :num_bus])
    qFilled = pd.DataFrame(power[:, num_bus:])

    return pFilled, qFilled

def fillValuesLRInverse(p, q, v, a):
    """
        Takes power and trains
        a linear regression model to fill in missing voltage and phase angle data.

        Does this by first filtering out all rows that can't be trained on.
        Then train the model on that data.
        Then fill in the missing data

    Input:
        p: the real power matrix
        q: the reactive power matrix
        v: the voltage matrix
        a: the phase angle matrix
    Ouput:
        vFilled: the real power matrix with missing values filled in
        aFilled: the reactive power matrix with missing values filled in
    """

    dataNonNull = nonNullIntersection([p, q, v, a])
    pNonNull = dataNonNull[0].values
    qNonNull = dataNonNull[1].values
    vNonNull = dataNonNull[2].values
    aNonNull = dataNonNull[3].values

    num_samples, num_bus = pNonNull.shape

    pq = np.zeros((num_samples, 2 * num_bus))
    pq[:, np.arange(0, num_bus)] = pNonNull
    pq[:, np.arange(num_bus, 2 * num_bus)] = qNonNull

    va = np.zeros((num_samples, 2 * num_bus))
    va[:, np.arange(0, num_bus)] = vNonNull
    va[:, np.arange(num_bus, 2 * num_bus)] = aNonNull

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaledX = scaler_x.fit_transform(pq)
    scaledY = scaler_y.fit_transform(va)

    model = LinearRegression().fit(scaledX, scaledY)

    power = pd.concat([p, q], axis=1).values
    voltage = pd.concat([v, a], axis=1).values

    for j in range(num_samples):
        if np.isnan(np.sum(voltage[j])) and not np.isnan(np.sum(power[j])):

            power_scaled = scaler_x.transform(power[j].reshape(1,-1))
            predictions = model.predict(power_scaled)

            for i in range(2 * num_bus):
                if np.isnan(voltage[j, i]):
                    pred = predictions[0]
                    scaled = scaler_y.inverse_transform(pred)[i]

                    power[j, i] = scaled

    vFilled = pd.DataFrame(voltage[:, :num_bus])
    aFilled = pd.DataFrame(voltage[:, num_bus:])

    return vFilled, aFilled
