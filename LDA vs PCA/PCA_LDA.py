# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 02:55:23 2017

@author: Mayuri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



marker_size = 7

def do_PCA(dataForPca):
    # get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca.T)
    
    # eigendecomposition of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(covarianceMatrix)
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    #sorting in descending order by eigenvalues
    eig_pairs.sort(reverse=True, key=(lambda x: x[0]))
    
    #pcaScores = np.matmul(pca_data, eig_vecs)
    
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1)))
    matrix_w2 = np.hstack((eig_pairs[1][1].reshape(3,1)))
    projection1 = np.matmul(dataForPca,matrix_w)
    projection2 = np.matmul(dataForPca,matrix_w2)
    print(np.var(projection1))
    print(np.var(projection2))
    print(eig_pairs[0][0])
    print(eig_pairs[1][0])
    
    #Question 7: 
    #Projection on PC1 : 158.364965864
    #Projection on PC2 : 5.25773450864
    #eigenvalues for PC1 : 161.049117827
    #eigenvalues for PC2 : 5.34684865286
    #Projections on PC1 and PC2 and eigenvalues for PC1 and PC2 are somewhat similar
        
    
    return eig_pairs,projection1
    

def do_LDA(X,y):
    mean_vectors = []
    #Calculating mean vector for each class
    for cl in range(0,2):
        mean_vectors.append(np.mean(X[y==cl], axis=0))
        
       
    #Computing Scatter Matrices
    S_WithinClass = np.zeros((2,2))                               # Within-class scatter matrix 
    for cl,meanvec in zip(range(0,2), mean_vectors):
        class_sc_mat = np.zeros((2,2))                            # scatter matrix for every class
        for row in X[y == cl]:
            row, meanvec = row.reshape(2,1), meanvec.reshape(2,1) # make column vectors
            class_sc_mat += (row-meanvec).dot((row-meanvec).T)
        S_WithinClass += class_sc_mat                               # sum class scatter matrices
    
    overall_mean = np.mean(X, axis=0)
    
    S_BetweenClass = np.zeros((2,2))                              # Between-class scatter matrix
    for i,mean_vec in enumerate(mean_vectors):  
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(2,1)                           # make column vector
        overall_mean = overall_mean.reshape(2,1)                   # make column vector
        S_BetweenClass += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    
    
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_WithinClass).dot(S_BetweenClass))
    
    # List of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        
    W = np.hstack((eig_pairs[0][1].reshape(2,1)))
    
      
    projection = np.matmul(X, W)
    #Question 8
    print('variance of the projections onto the W axis',np.var(projection))
    #5.22507522401
    
    return  projection,W
# observations
    
def main():
    
    in_file_name = 'dataset_1.csv'
    data_in = pd.read_csv(in_file_name)
    AnalysisData=data_in.as_matrix()
    eig_pairs, projection1 = do_PCA(AnalysisData)
    projection,W = do_LDA(AnalysisData[:,:2],AnalysisData[:,2])
    
    
    #plots  
    #Question 1: We can see clear seperation in raw data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(AnalysisData[:,0], AnalysisData[:,1], label='V2 vs V1')

    
    #Question 2: Do you still see a clear separation of the data in PC1, i.e. in
    #projections of your raw data on the PC1 axis?
    #There is clear sepearion of raw data on PC1 axis
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(AnalysisData[:,0], AnalysisData[:,1], label='V2 vs V1')
    plt.plot([0, 50*eig_pairs[0][1][0]], [0, 50*eig_pairs[1][1][1]],
                color='r', linewidth=3, label='PC1')
    plt.xlabel('V1')
    plt.ylabel('V2')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(projection1[0:30], np.zeros(30), linestyle='None', marker='o', markersize=marker_size, color='blue', label='class 1')
    plt.plot(projection1[30:60], np.zeros(30), linestyle='None', marker='o', markersize=marker_size, color='red', label='class 0')
    plt.title('Projection of PC1')
    plt.xlabel('projection')
    plt.ylabel('')
    plt.legend()
    
    #Question 4: W is obtained after LDA is applied
   
    
    #Question 5: clear separation of the data in the projection onto W is seen
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title('Results from applying LDA to  data')
    plt.xlabel('projection')
    plt.ylabel('')
    plt.plot(projection[0:30], np.zeros(30), linestyle='None', marker='o', markersize=marker_size, color='blue', label='class 1')
    plt.plot(projection[30:60], np.zeros(30), linestyle='None', marker='o', markersize=marker_size, color='red', label='class 0')
    plt.legend()
     
    #Question 6:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    W_scaled = W * 12.0 / W[0]
    plt.scatter(AnalysisData[:,0], AnalysisData[:,1], label='V2 vs V1')
    plt.plot([0, 50*eig_pairs[0][1][0]], [0, 50*eig_pairs[1][1][1]],
                color='r', linewidth=3, label='PC1')
    plt.plot([0, W_scaled[0]], [0, W_scaled[1]], color='green')
    plt.title('PC1 and W axis')
    plt.xlabel('V1')
    plt.ylabel('V2')

    #Question 9: 
    #PCA is unsupervised and LDA is supervised
    #PCA projected axes are selected based on maximum variance 
    #LDA axes are selected in such a way that the interclass distance is maximized whereas the intraclass distance is minimized
if __name__ == "__main__":
    main()    

