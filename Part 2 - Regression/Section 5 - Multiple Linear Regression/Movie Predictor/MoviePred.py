import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('movie_data.csv')

x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, :1].values

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x , y ,test_size = 0.2, random_state = 0 )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

ypredictor = regressor.predict(xtest)
print ("Primary Predictor")
print (ypredictor)


print ("Primary Test Values")
print (ytest)

#Backward Elimination ... considering the significant level 0.02

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((10,1)).astype(int),values = x , axis= 1)


x_opt = x[:,[0,1,2,3]]
regressor_ols= sm.OLS(endog = y, exog = x_opt).fit()
summary = regressor_ols.summary()


x_opt = x[:,[1,2,3]]
regressor_ols= sm.OLS(endog = y, exog = x_opt).fit()
summary = regressor_ols.summary()


x_opt = x[:,[1,2]]
regressor_ols= sm.OLS(endog = y, exog = x_opt).fit()
summary2 = regressor_ols.summary()



from sklearn.model_selection import train_test_split
xtrain2 , xtest2 , ytrain2 , ytest2 = train_test_split(x_opt , y ,test_size = 0.2, random_state = 0 )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain2,ytrain2)



final_movie_pred = regressor.predict(xtest2) 
print ("Final Predictor")
print (final_movie_pred)



print ("Final Test Values")
print (ytest2)