# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 02:38:42 2017

@author: Mayuri
"""
import datetime
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

def neuralnetwork(x,y,iterations):
    #Variable initialization
    #weight initialization
    inputlayer_neurons = x.shape[1] #number of features in data set
    hiddenlayer_neurons = 2 #number of hidden layers neurons
    output_neurons = 1 #number of neurons at output layer
    theta1 = np.random.rand(hiddenlayer_neurons, inputlayer_neurons+1)
    theta2 = np.random.rand(output_neurons, hiddenlayer_neurons+1)
    y = np.matrix(y)
    m = x.shape[0]
    for i in range(iterations):
        a1 = np.concatenate((np.ones((99,1)), x), axis=1)  #bias initialization
        z2 = np.matmul(theta1,a1.transpose())
      
        a2 = sigmoid(z2)
        a2 = np.insert(a2,0,1,axis = 0)  #bias initialization 
        
        z3 = np.matmul( theta2,a2)
        
        a3 = sigmoid(z3)
        
        delta_3 = a3-y
      
        grad = np.multiply(a2,(1-a2))
        abc = theta2.transpose()*delta_3
        delta_2 = np.multiply(abc,grad)
   
        delta_2 = delta_2[1:,:]
        
        D1 = delta_2*a1/m
    
        D2 = delta_3*a2.transpose()/m
      
        #cost = y * np.log10(a3) + (1.0-y) * np.log10(1.0 - a3)

        theta1 = theta1 - 10*D1/m
        theta2 = theta2 - 10*D2/m               
    return theta1,theta2

def predict(x,theta1,theta2):
    d = np.matrix(x)
    
    #Forward Propogation  
    a1 = np.concatenate((np.ones((1,1)), d), axis=1)  #bias initialization   
    z2 = np.matmul(theta1,a1.transpose())
    a2 = sigmoid(z2)
    a2 = np.insert(a2,0,1,axis = 0)  #bias initialization  
    z3 = np.matmul( theta2,a2)   
    y_pred_probability = sigmoid(z3)
  
    if y_pred_probability >= 0.5:
        y_pred_binary = 1.0
    else:
        y_pred_binary = 0.0
    return y_pred_binary       
      

def main():
    #Input array
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target    
    data = X[y!=0,2:]  #Considering petal length and petal width from the dataset.
    iterations = 10000
    for i in range(data.shape[0]):    #scaling
        data[i,0]= (data[i,0]-np.min(data[:,0]))/(np.max(data[:,0])-np.min(data[:,0]))
        data[i,1]= (data[i,1]-np.min(data[:,1]))/(np.max(data[:,1])-np.min(data[:,1]))
    target = y[:100]    #Considering target variable as 0 for virginica and 1 for versicolor flowers
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
        theta1,theta2 = neuralnetwork(X_train,y_train, iterations)
        
        predicted_y = predict(X_test,theta1,theta2)
        
        if (predicted_y == y_test):
            accuracy = accuracy + 1
        
    average_error_rate = 1 - (accuracy/data_for_analysis.shape[0])
    print('Average error rate',average_error_rate)    
    print('Accuracy',accuracy)
    print(datetime.datetime.now())
    

if __name__ == "__main__":
    main()    
