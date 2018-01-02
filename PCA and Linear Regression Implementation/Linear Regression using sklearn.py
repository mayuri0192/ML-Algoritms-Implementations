# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:16:34 2017

@author: Mayuri Deshpande
Student Id: 800972831
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


def do_linearRegression(diabetes_data,diabetes_target):
    # Split the data into training/testing sets
    diabetes_data_train, diabetes_data_test, diabetes_target_train, diabetes_target_test = train_test_split(diabetes_data, diabetes_target, test_size=20, random_state=None)
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(diabetes_data_train, diabetes_target_train)
    
    # Make predictions using the testing set
    diabetes_target_pred = regr.predict(diabetes_data_test)
    
    # Collect results
    regrResults = { 'train_x': diabetes_data_train,
                   'train_y': diabetes_target_train,
                   'test_x': diabetes_data_test,
                   'test_y': diabetes_target_test,
                   'pred_y': diabetes_target_pred,
                   'regr': regr}
    return regrResults
 
def main():
    
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()
    
    # one feature is used
    diabetes_data = diabetes.data
    diabetes_data = diabetes_data[:,np.newaxis, 2]
    diabetes_target = diabetes.target
    results = do_linearRegression(diabetes_data,diabetes_target)
    
    # The coefficients
    print('Coefficients: \n', results['regr'].coef_)
     
    # Plot outputs
    plt.scatter(results['test_x'], results['test_y'],  color='b', label='testing x vs testing y')
    plt.plot(results['test_x'],  results['pred_y'], color='g', linewidth=3, label='testing x vs predicted y')    
    plt.title('Linear Regression on the diabetes dataset')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()
    
