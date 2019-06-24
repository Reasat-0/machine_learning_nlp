import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset.....
dataset = pd.read_csv('Position_Salaries.csv')  
x = dataset.iloc[:, 1:2].values  #here 1:2 the number 2 is to make the x,,,, a matrix
y = dataset.iloc[:, 2].values

#In this we dont need to split the dataset into Training set and data set ... cause here data are less...we would need more

#Fitting the linear regression model to our dataset ... actually these iis to compare with the polynomial...
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(x,y)
linPrediction = linReg.predict(x)

#Fitting the polynomial regression to our dataset.....
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4)
        #to store the degree 2(x^2) we need to create a new matrix
x_mat = polyReg.fit_transform(x)
#print (x_mat)


#now we have to fit the linear regression model to the x_mat as the linear regression model is on behalf of polynomial regression
linReg2 = LinearRegression()
linReg2.fit(x_mat,y)
polyPrediction = linReg2.predict(x_mat)


#visualising the Linear MOdel 
plt.title('Truth or Bluff')
plt.scatter(x,y,color= 'red')
#plt.plot(x,linPrediction, color = 'green')
plt.xlabel('Postition')
plt.ylabel('Salary')


#visualaising polynomial model
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(-1,1)

plt.scatter(x,y,color = 'red')
plt.plot(x_grid,linReg2.predict(polyReg.fit_transform(x_grid)))


#to predict a single value .... 
