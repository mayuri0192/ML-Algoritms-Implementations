Implementation of PCA and LDA in Python

Dataset: dataset_1.csv, columns correspond to variables and there are two variables named V1 and V2.

Questions: (All questions answered in PCA_LDA.py)

(1) Plot V2 vs V1. Do you see a clear separation of the raw data?

(2) Apply your own PCA function to this dataset without scaling the two variables. 
Project the raw data onto your first principal component axis, i.e. the
PC1 axis. Do you still see a clear separation of the data in PC1, i.e. in
projections of your raw data on the PC1 axis?

(3) Add the PC1 axis to the plot you obtained in (1).

(4) Apply your own LDA function to this dataset and obtain W. The class
information of each data point is in the label column.

(5) Project your raw data onto W. Do you see a clear separation of the data in
the projection onto W?

(6) Add the W axis to your plot. At this point, your plot should contain the raw
data points, the PC1 axis you obtain from the PCA analysis, and the W axis
you obtain from the LDA analysis.

(7) Compute the variance of the projections onto PC1 and PC2 axes. What
is the relationship between these two variances and the eigenvalues of the
covariance matrix you use for computing PC1 and PC2 axes?

(8) Compute the variance of the projections onto the W axis.

(9) What message can you get from the above PCA and LDA analyses?
