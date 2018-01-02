# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:07:48 2017

@author: Mayuri
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

marker_size = 7


def do_LDA(X,y):
    mean_vectors = []
    #Calculating mean vector for each class
    for cl in range(0,2):
        mean_vectors.append(np.mean(X[y==cl], axis=0))
       
    #Computing Scatter Matrices
    S_WithinClass = np.zeros((19,19))                               # Within-class scatter matrix 
    for cl,meanvec in zip(range(0,2), mean_vectors):
        class_sc_mat = np.zeros((19,19))                            # scatter matrix for every class
        for row in X[y == cl]:
            row, meanvec = row.reshape(19,1), meanvec.reshape(19,1) # make column vectors
            class_sc_mat += (row-meanvec).dot((row-meanvec).T)
        S_WithinClass += class_sc_mat                               # sum class scatter matrices
    
    overall_mean = np.mean(X, axis=0)
    
    S_BetweenClass = np.zeros((19,19))                              # Between-class scatter matrix
    for i,mean_vec in enumerate(mean_vectors):  
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(19,1)                           # make column vector
        overall_mean = overall_mean.reshape(19,1)                   # make column vector
        S_BetweenClass += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    
    
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_WithinClass).dot(S_BetweenClass))
    
    # List of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        
    W = np.hstack((eig_pairs[0][1].reshape(19,1)))
    print('Matrix W:\n', W.real)
      
    projection = np.matmul(X, W)
    return  projection
  
def do_LDA_skLearn(X,y): 
    # apply sklearn LDA to dataset
    sklearn_LDA = LDA(n_components=2)
    sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)
    return sklearn_LDA_projection
    
def main():
    
    in_file_name = "SCLC_study_output_filtered_2.csv"
    data_in = pd.read_csv(in_file_name, index_col=0)
    X = data_in.as_matrix()
    y = np.concatenate((np.zeros(20), np.ones(20)))
    
    II_0 = np.where(y==0)
    II_1 = np.where(y==1)
    
    II_0 = II_0[0]
    II_1 = II_1[0]
    
    projection = do_LDA(X,y)
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title('Results from applying LDA to cell line data')
    plt.xlabel('projection')
    plt.ylabel('')
    plt.plot(projection[0:20], np.zeros(20), linestyle='None', marker='o', markersize=marker_size, color='blue', label='NSCLC')
    plt.plot(projection[20:40], np.zeros(20), linestyle='None', marker='o', markersize=marker_size, color='red', label='SCLC')
    plt.legend()
    
    plt.show()
    
    sklearn_LDA_projection = do_LDA_skLearn(X,y)
    
    
    # plot the projections
    plt.figure()
    
    plt.title('Results from applying sklearn LDA to cell line data')
    plt.xlabel('sklearn_LDA_projection')
    plt.ylabel('')
    plt.plot(sklearn_LDA_projection[II_0], np.zeros(len(II_0)), linestyle='None', marker='o', markersize=marker_size, color='blue', label='NSCLC')
    plt.plot(sklearn_LDA_projection[II_1], np.zeros(len(II_1)), linestyle='None', marker='o', markersize=marker_size, color='red', label='SCLC')
    plt.legend()
    
    plt.show()
    
    ####################################################################################
    #Plots are similar but the projections obtained are different
if __name__ == "__main__":
    main()    
