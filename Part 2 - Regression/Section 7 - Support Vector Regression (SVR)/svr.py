import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset.....
dataset = pd.read_csv('Position_Salaries.csv')  
x = dataset.iloc[:, 1:2].values  #here 1:2 the number 2 is to make the x,,,, a matrix
y = dataset.iloc[:, 2].values

"""
from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x , y ,test_size = 0.2, random_state = 0 )
"""
#featuring scaling of data as SVR doesnt include feature scaling in it...so we have to do it for bttr result

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)


y = y.reshape(10,-1)


'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)
'''



#Fitting the regression model SVR to our dataset.....
from sklearn.svm import SVR
svrRegressor = SVR(kernel = 'rbf')
svrRegressor.fit(x,y)
ypred = svrRegressor.predict([[6.5]])


#visualising the Linear MOdel 
plt.title('Truth or Bluff')
plt.scatter(x,y,color= 'red')
plt.plot(x,svrRegressor.predict(x),color ='green')
plt.xlabel('Postition')
plt.ylabel('Salary')
plt.show();

#for higher points and smoother curve
'''
xgrid = np.arange(min(x),max(x),0.1)
xgrid = xgrid.reshape((len(xgrid),1))
plt.title('Truth or Bluff')
plt.scatter(x,y,color= 'red')
plt.plot(xgrid,regressor.predict(xgrid),color= 'green')
plt.xlabel('Postition')
plt.ylabel('Salary')




plt.scatter(x,y,color = 'red')
plt.plot(x,regressor.predict(x))


#to predict a single value .... 
 '''