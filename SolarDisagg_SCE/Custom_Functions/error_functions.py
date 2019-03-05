import numpy as np 
import warnings

#	returns the Mean Percentage Error between two Numpy Arrays. The two lists must be of the same size and greater than 0.
def mpe(prediction, actual):
	
	try:
		return np.mean((prediction - actual) / actual)
	except:
		return 'NA'

#	returns the Coefficient of Variation between two Numpy Arrays. The two lists must be of the same size and greater than 0.
def cv(prediction, actual): 
	return rmse(prediction, actual)/np.mean(actual)

#   returns the Root-Mean-Square Error between two Numpy Arrays. The two lists must be of the same size and greater than 0.
def rmse(prediction, actual):
	return np.sqrt(((prediction - actual) ** 2).mean()) 

#   returns the Mean Absolute Percentage Error between two Numpy Arrays. The two lists must be of the same size and no values of 0 in Actual list.
def mape(prediction, actual):
	try:
		return np.mean(np.abs((actual - prediction) / actual)) 
	except:
		return 'NA'
       
    
def MAPE_pos(prediction,actual):
    eva = np.abs(actual) > (0.05 * np.abs(np.mean(actual)))
    return np.mean(np.abs((actual[eva] - prediction[eva]) / actual[eva]))

def cv_pos(prediction,actual):
    eva = np.abs(actual) > (0.05 * np.mean(np.abs(actual)))
    return np.sqrt(np.mean((prediction[eva]-actual[eva])**2))/np.mean(actual[eva])

def rmse_pos(prediction,actual):
    eva = np.abs(actual) > (0.05 * np.mean(np.abs(actual)))
    if sum(eva) == 0: # If the actual is always zero, then evaluate on the prediction
        eva = np.abs(prediction) > (0.05 * np.mean(np.abs(prediction)))
    return np.sqrt(np.mean((prediction[eva]-actual[eva])**2))
 
def mpe_pos(prediction,actual):
    eva = np.abs(actual) > (0.05 * np.abs(np.mean(actual)))
    return mpe(prediction[eva],actual[eva])
    
def getErrors(prediction, actual):

	mpe_err = mpe(prediction, actual)
	mape_err = mape(prediction,actual)
	cv_err = cv(prediction,actual)
	
	
	print("MPE is",mpe_err)
	print("MAPE is",mape_err)
	print("CV is",cv_err)
	# print("RMSE is",rmse_err)
	
	mpe_err = str(mpe_err)[0:10]
	mape_err = str(mape_err)[0:10]
	cv_err = str(cv_err)[0:10]

	# print(mpe_err, type(mpe_err))

	return [mpe_err,mape_err,cv_err]
