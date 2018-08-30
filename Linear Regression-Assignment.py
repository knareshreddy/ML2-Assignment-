# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:04:00 2018

@author: Naresh Kumar
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
boston = load_boston()
bos= pd.DataFrame(boston.data)
bos.columns=boston.feature_names

print(boston.DESCR)

#Defining X and y datasets 

X=bos.values
y=boston.target


#Scaling the data
X=sklearn.preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train) #  MODEL can predict woith accuracy of 76.76 %
regressor.coef_
regressor.intercept_


# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Optimization of Model

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
X_opt=X[:,:]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=np.delete(X, [3,7], axis=1) # as Columns 3 and 7 are shown insignificant in 
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


# plot
plt.scatter(regressor.predict(X_train),regressor.predict(X_train) - y_train,c='b', s=40, alpha=0.5)
plt.scatter(regressor.predict(X_test),regressor.predict(X_test) - y_test, c='g', s=40)
plt.hlines(y=0,xmin=0,xmax=50)      # Horizontal line
plt.show()
