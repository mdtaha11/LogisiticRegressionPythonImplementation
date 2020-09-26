# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:30:38 2020

@author: Taha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv("ex1data1.txt")
X=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values

m=len(X)
plt.scatter(X,y)

theta=0
alpha=0.01

X=X[:,np.newaxis]
y=y[:,np.newaxis]
iterations=1500 
theta=np.zeros([2,1])
ones = np.ones((m,1))
X=np.hstack((ones ,X))


z=np.transpose(X)*theta

def sigmoid(z):
    g=1/(1+np.exp(-z))

 
def computeCost(X,y,theta):
    temp=np.dot(X,theta)-y
    return np.sum(np.power(temp,2))/(2*m)

J=computeCost(X,y,theta)

def gradDesc(X,y,theta,alpha,iterations):
    for _ in range(iterations):
        temp=np.dot(X,theta)-y
        temp=np.dot(X.T,temp)
        theta =theta-(alpha/m)*temp
    return theta

theta=gradDesc(X,Y,theta,alpha,iterations)

J=computeCost(X,y,theta)

plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()