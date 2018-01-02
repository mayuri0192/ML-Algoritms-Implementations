# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:51:20 2017

@author: Mayuri Deshpande
Student Id: 800972831
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
def do_PCA(dataForPca):
    # get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca.T)
    
    # eigendecomposition of the covariance matrix
    eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
    II = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[II]
    eigenVectors = eigenVectors[:, II]
    
    # plotting Principle Component
    plt.plot([0, -5*eigenVectors[0,0]], [0, -5*eigenVectors[1,0]],
            color='r', linewidth=3, label='PC1')
    plt.legend()
    plt.show()

def estimate_coef(x, y): 
    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)
 
    # calculating cross-deviation and deviation about x    
    SS_xy = np.sum((y - m_y) * (x - m_y))
    SS_xx = np.sum((x - m_x)**2)
 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return(b_0, b_1)
 
def plot_regression_line(x, y, y_t, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30, label='y vs x')
 
    # plotting the actual points as scatter plot
    plt.scatter(x, y_t, color = "b",
               marker = "o", s = 30, label='y-theoretical vs x')
 
    # predicted response vector
    y_pred = b[0] + b[1]*x
 
    # plotting the regression line
    plt.plot(x, y_pred, color = "g", linewidth=3, label='Regression Line')

def main():
    # observations
    in_file_name = 'linear_regression_test_data.csv'
    data_in = pd.read_csv(in_file_name)
    AnalysisData=data_in.as_matrix()
    AnalysisData=AnalysisData[:,1:]
    x = data_in['x']
    y = data_in['y']
    y_t = data_in['y_theoretical']    

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
 
    plt.title('Raw data, PC axis and Regression Line')
    # plotting regression line
    plot_regression_line(x, y, y_t, b)
    
    #Perform PCA
    do_PCA(AnalysisData)
    #PC1 axis and the regression line are very similar
    
if __name__ == "__main__":
    main()
