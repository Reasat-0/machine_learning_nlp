import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Dataset

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Label Encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncObject = LabelEncoder()
x[:,3] = LabelEncObject.fit_transform(x[:,3]) #here it is turnned into numbers...

OneHotObject = OneHotEncoder(categorical_features = [3])
x = OneHotObject.fit_transform(x).toarray()

#To ignore the Dummy Variable Trap...though python LInear Regression library take care of this...i jst use it for exprnc purpose

x = x[:, 1:] #here we are taking all the columns of x but starting from index 1.so its ignoring the index 0....

x= x.astype(int)

#Splitting into training and test set

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain, ytest = train_test_split(x , y ,test_size = 0.2, random_state = 0)

#Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

ypred = regressor.predict(xtest)
print ("Prediction 1  - ")
print (ypred)

#Backward Elimination 
import statsmodels.formula.api as sm


#now we need to add a new variable x0 withe b0 of our eqn Y= b0x0 + b1x1 + b2x2 .... +bnxn where x0 will be 1

x = np.append(arr = np.ones((50,1)).astype(int),values = x , axis = 1)

#Backward elimination step - 2
x_optimal = x[:,[0,1,2,3,4,5]]

#Backward elimination step - 3
regressor_ols = sm.OLS(endog= y , exog= x_optimal).fit()    #[[ols is a class..Means "Ordinary Least squars" ... parameter endog means the                                                                   dependent and exog means the dependnt variable]]
sumar = regressor_ols.summary() #here as a result we get the highest p value in column with index 2. so next we will exclude this

#Backword El step-4 *** Removing the predictor
x_optimal = x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y ,exog = x_optimal).fit()
summary2 = regressor_ols.summary()

x_optimal = x[:, [0,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = x_optimal).fit()
summary3 = regressor_ols.summary()

x_optimal = x[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = x_optimal).fit()
summary4 = regressor_ols.summary();

x_optimal = x[:,[0,3]]
regressor_ols = sm.OLS(endog = y, exog = x_optimal ).fit()
summary5 = regressor_ols.summary()





from sklearn.model_selection import train_test_split
xtrain2 , xtest2 , ytrain2 , ytest2 = train_test_split(x_optimal , y ,test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain2,ytrain2)

ypred2 = regressor.predict(xtest2)

print ("Prediction 2  - ")
print (ypred2)


print ("Y test - ")
print (ytest2)





