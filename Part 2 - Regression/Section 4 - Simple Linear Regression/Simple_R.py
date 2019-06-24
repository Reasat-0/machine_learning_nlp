#Importing Libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

#Spliting the dataset into Training and Test set

from sklearn.model_selection import train_test_split    #the model_selection replaced the cross validation 
xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size = 1/3 , random_state = 0)

#Model Building

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

#Prediction using the model

ypred = regressor.predict(xtest)

#Visualizing the Models prediction
#training set
plt.scatter(xtrain,ytrain , color= 'red') #scatter makes the graph with points
plt.plot(xtrain,regressor.predict(xtrain), color= 'green')
plt.title('Salary vs Age training')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#test set

plt.scatter(xtest,ytest , color= 'red') #scatter makes the graph with points
plt.plot(xtrain,regressor.predict(xtrain), color= 'blue')
plt.title('Salary vs Age (test set) ')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
