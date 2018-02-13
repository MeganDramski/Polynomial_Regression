#!/usr/bin/env python2
# -*- coding: utf-8 -*-




import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset=pd.read_csv('Position_SAlaries.csv')
X = dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values


# Spliting the dataset into the Traing set and Test set 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.3, random_state=0)



#Fitting the decision tree regression to the dataset 
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


#Predicting
y_pred= regressor.predict(6.5)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid= X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='green')
plt.title('truth or bluff (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

