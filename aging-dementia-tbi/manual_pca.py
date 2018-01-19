'''PCA script:
Manual Implemention of PCA using
NumPy Library
---------------
author: Jeremy Grace
email: jeremy.grace@outlook.com

'''

import numpy as np


def perform_pca(X):
    '''Calculate the principal components for
    a given data matrix = X - and return transformed
    matrix into a new v-dimensional subspace
    '''
    # standardize the data matrix
    X_norm = (X - np.mean(X, axis=0)/np.std(X, axis=0))
    # calculate the convariance matrix
    C = np.cov(X_norm.T)
    # retrieve the eignvalues and eignvectors
    eig_vals, eig_vecs = np.linalg.eig(C)
    # pair up the eignvectors with corresponding eignvalues
    eig = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig.sort(reverse=True)
    # dimensional reduce by a built projection matrix and
    # transforming the initial data into a new v-dimensional subspace
    Y = np.hstack((eig[0][1][:, np.newaxis], eig[1][1][:, np.newaxis]))
    X_pca = np.dot(X_norm, Y)

    return X_pca


if __name__ == "__main__":
    X = input("Enter data matrix for PCA: ")
    perform_pca(X)
