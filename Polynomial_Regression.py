#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:41:35 2018

@author: meganpolak
"""



import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

#Fitting Linear Regresion to the dataset 
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)


#Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing  import PolynomialFeatures 
poly_reg= PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2= LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing the Linear Regression results 
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='yellow')
plt.title('truth or bluf(Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


#Visualizing the Polynomial Regression results 
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='yellow')
plt.title('truth or bluff (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
