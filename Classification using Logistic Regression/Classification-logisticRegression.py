# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:29:11 2017

@author: Mayuri
"""
import datetime
from sklearn import datasets

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

    

def logisticRegression(x,y):
    #Variable initialization
    intercept = np.ones((x.shape[0], 1))
    x = np.hstack((intercept, x))
    features = x.shape[1] #number of features in data set
    
    #weight initialization
    w = np.zeros(features)
    theta = np.reshape(w,(len(w),1))
    
    iteration = 200
    
    theta = np.zeros((iteration, features))    
    initial_theta =  np.random.uniform(features)    
    theta[0, :] = initial_theta
    m = 99
    alpha = 0.9   #Setting learning rate
    
    J = np.zeros(iteration)       
    for index_iter in range(iteration-1):
        partial_derivative = np.zeros(x.shape[1])
    
        for i in range(m):
            current_z = sum(x[i, :] * theta[index_iter, :])
            
            current_y_hat = sigmoid(current_z)
            current_residual = current_y_hat - y[i]
            partial_derivative = partial_derivative + x[i, :] * current_residual
    
            current_cost = y[i] * np.log10(current_y_hat) + (1.0-y[i]) * np.log10(1.0 - current_y_hat)
            J[index_iter] = J[index_iter] + current_cost
    
        J[index_iter] = -J[index_iter] / m
        
        
        theta[index_iter+1, :] = theta[index_iter, :] - alpha * partial_derivative / m
        
    return theta

def predict(X_test,theta):
# prediction
    
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.hstack((intercept, X_test))
    final_theta = theta[-1, :]
    final_theta = final_theta.reshape((len(final_theta), 1))  
    z_pred = np.matmul(X_test, final_theta)
    y_pred_probability = sigmoid(z_pred)    
    y_pred_binary = np.zeros(len(y_pred_probability))
    for i in range(len(y_pred_probability)):
        if y_pred_probability[i] >= 0.5:
            y_pred_binary[i] = 1.0
        else:
            y_pred_binary[i] = 0.0
    return y_pred_binary       


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target    
    data = X[y!=0,2:]
    for i in range(data.shape[0]):
        data[i,0]= (data[i,0]-np.min(data[:,0]))/(np.max(data[:,0])-np.min(data[:,0]))
        data[i,1]= (data[i,1]-np.min(data[:,1]))/(np.max(data[:,1])-np.min(data[:,1]))
    target = y[:100]
    target = np.reshape(target,(len(target),1))
    data_for_analysis = np.hstack((data,target))
    accuracy = 0;
    for i in range(data_for_analysis.shape[0]):
        dataIn = data_for_analysis
        
        X_test = data_for_analysis[i,:2]
        X_test = np.matrix(X_test)
        y_test = data_for_analysis[i,2]
        dataIn = np.delete(dataIn,i,0)
        
        X_train = dataIn[:,:2]
        y_train = dataIn[:,2]
        theta = logisticRegression(X_train,y_train)
        
        predicted_y = predict(X_test,theta)
        if (predicted_y == y_test):
            accuracy = accuracy + 1
    average_error_rate = 1 - (accuracy/data_for_analysis.shape[0])
    print('Average error rate',average_error_rate)
    print('Accuracy',accuracy)
    print(datetime.datetime.now())
    

if __name__ == "__main__":
    main()    
