import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, [2,3] ].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
xtrain = sc_X.fit_transform(xtrain)
xtest = sc_X.fit_transform(xtest)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain,ytrain)

#Predicting on the test set

ypred = classifier.predict(xtest)

#confusion matrics generation

from sklearn.merics import confusion_matrix
confusion = confusion_matrix(ytest,ypred)
