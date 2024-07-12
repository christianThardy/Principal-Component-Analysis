# Needs refactoring
# Dependencies
import timeit

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from ipywidgets import interact

from load_data import load_mnist

# Image data
MNIST = load_mnist()
images, labels = MNIST['data'], MNIST['target']

%matplotlib inline

# Quick plot of an image from the dataset to make sure data loaded properly
plt.figure(figsize = (4,4))
plt.imshow(images[0].reshape(28,28), cmap = 'gray');


'''
   The next four functions preprocess the data so the images can have a zero mean and variance of one
   The pixel encodings are originally uint8 and must be converted to a floating point number between 0-1
   The mean μ must be subtracted from each image
   Each image must be scaled dimensionally by 1/σ, where σ is the standard deviation
   Xbar will be the normalized dataset
'''
def normalize(X):
    '''
        In the future, you could encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those 
        dimensions when doing normalization.
    '''
    mu = np.mean(X, axis = 0) 
    std = np.std(X, axis = )
    std_filled = std.copy()
    std_filled[std == 0] = 1.
    Xbar = (X - mu) / std_filled                  
    return Xbar, mu, std


# Computes the eigenvalues and corresponding eigenvectors for the covariance matrix S and sorts by the 
# largest eigenvalues and corresponding eigenvectors
def eig(S):
    ''' 
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    '''
    eigen_values, eigen_vectors = np.linalg.eig(S)
    idx = eigen_values.argsort()[:: - 1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    return (eigen_values, eigen_vectors) 


# Computes the projection matrix onto the space spanned by `B` where:
# B = ndarray of dimension (D, M), the basis for the subspace and returns P = projection matrix
def projection_matrix(B):
    return B @ (np.linalg.inv(B.T @ B)) @ B.T
 

'''
   X = ndarray of size (N, D), where D is the dimension of the data,
   and N is the number of datapoints
   num_components = the number of principal components to use
   X_reconstruct: ndarray of the reconstruction of X from the first `num_components` principal components
'''
def PCA(X, num_components):
    X, mean, std = normalize(X)
    S = np.cov(X, rowvar = False, bias = True)
    eig_vals, eig_vecs = eig(S)
    P = projection_matrix(eig_vecs[:, :num_components])
    X_reconstruct = (P @ X.T).T
    return X_reconstruct 


# How many principal components do we need in order to reach a mean squared error of less than 100 for the dataset?
 def mse(predict, actual):
    '''Helper function that computes the mean squared error (MSE)'''
    return np.square(predict - actual).sum(axis = 1).mean()


'''
   PCA for high dimensional datasets

   When the dimensionality of the dataset is larger than the number of given samples, 
   PCA can be implemented in a more optimized way for high-dimensionality

   Computes PCA for small a sample size but high-dimensional features
   X = ndarray of size (N, D), where D is the dimension of the sample,
   and N is the number of samples
   num_components = the number of principal components to use
   X_reconstruct = (N, D) ndarray
   The reconstruction of X from the first `num_components` pricipal components
'''
def PCA_high_dim(X, n_components):
    N, D = X.shape
    M = (X @ X.T) / N
    eig_vals, eig_vecs = eig(M)
    U = (X.T @ eig_vecs)[:, :n_components]
    P = projection_matrix(U)
    X_reconstruct = (P @ X.T).T
    return X_reconstruct 


# Defines a function that finds time/space complexity by comparing the running time between PCA and PCA_high_dim
def time(f, repeat=10):
   times = []
   for _ in range(repeat):
      start = timeit.default_timer()
      f()
      stop = timeit.default_timer()
      times.append(stop-start)
   return np.mean(times), np.std(times)


# More data preprocessing
NUM_DATAPOINTS = 1000
X = (images.reshape(-1, 28 * 28)[:NUM_DATAPOINTS]) / 255.
print(X.shape)
Xbar, mu, std = normalize(X)
print(mu.shape)
print(std.shape)

# The number of principal components we use, the smaller the reconstruction error will be
for num_component in range(1, 20):
    from sklearn.decomposition import PCA as SKPCA
    # Computes a standard solution given by scikit-learn's implementation of PCA
    pca = SKPCA(n_components = num_component, svd_solver = 'full')
    sklearn_reconst = pca.inverse_transform(pca.fit_transform(Xbar))
    reconst = PCA(Xbar, num_component)
    np.testing.assert_almost_equal(reconst, sklearn_reconst)
    print(np.square(reconst - sklearn_reconst).sum())

# Loss and reconstruction variables
loss = []
reconstructions = []

# Iterates over different number of principal components, and computes the MSE
for num_component in range(1, 100):
    reconst = PCA(Xbar, num_component)
    error = mse(reconst, Xbar)
    reconstructions.append(reconst)
    # print('n = {:d}, reconstruction_error = {:f}'.format(num_component, error))
    loss.append((num_component, error))

reconstructions = np.asarray(reconstructions)
reconstructions = reconstructions * std + mu # 'unnormalize' the reconstructed image
loss = np.asarray(loss)

# Creates a table showing the number of principal components and MSE
pd.DataFrame(loss).head(10)

# A plot of the numbers from the table above
fig, ax = plt.subplots()
ax.plot(loss[:,0], loss[:,1]);
ax.axhline(100, linestyle = '--', color = 'r', linewidth=2)
ax.xaxis.set_ticks(np.arange(1, 100, 5));
ax.set(xlabel =' num_components', ylabel = 'MSE', title='MSE vs number of principal components');

# Invariant to test the PCA_high_dim implementation
 np.testing.assert_almost_equal(PCA(Xbar, 2), PCA_high_dim(Xbar, 2))
    
# Variables for time function
times_mm0 = []
times_mm1 = []

# Iterate over datasets of different size by computing the running time of X^TX and XX^T
for datasetsize in np.arange(4, 784, step = 0):
    XX = Xbar[:datasetsize] # Selects the first `datasetsize` samples in the dataset
    # Records the running time for computing X.T @ X
    mu, sigma = time(lambda : XX.T @ XX)
    times_mm0.append((datasetsize, mu, sigma))
    
    # Records the running time for computing X @ X.T

    mu, sigma = time(lambda : XX @ XX.T)
    times_mm1.append((datasetsize, mu, sigma))
    
times_mm0 = np.asarray(times_mm0)
times_mm1 = np.asarray(times_mm1)

# Plots the running time of computing X @ X.T(X^TX) and X @ X.T(XX^T)
fig, ax = plt.subplots()
ax.set(xlabel = 'size of dataset', ylabel = 'running time')
bar = ax.errorbar(times_mm0[:, 0], times_mm0[:, 1], times_mm0[:, 2], label = '$X^T X$ (PCA)', linewidth = 2)
ax.errorbar(times_mm1[:, 0], times_mm1[:, 1], times_mm1[:, 2], label = '$X X^T$ (PCA_high_dim)', linewidth = 2)
ax.legend();

%time Xbar.T @ Xbar
%time Xbar @ Xbar.T
pass # Put this here so that our output does not show result of computing `Xbar @ Xbar.T`

# Iterate over datasets of different size and benchmarks the running time of both algorithms
times0 = []
times1 = []

for datasetsize in np.arange(4, 784, step = 100):
    XX = Xbar[:datasetsize]
    npc = 2
    mu, sigma = time(lambda : PCA(XX, npc), repeat = 10)
    times0.append((datasetsize, mu, sigma))
    
    mu, sigma = time(lambda : PCA_high_dim(XX, npc), repeat = 10)
    times1.append((datasetsize, mu, sigma))
    
times0 = np.asarray(times0)
times1 = np.asarray(times1)

# Plots the time/space complexity of each algorithm
fig, ax = plt.subplots()
ax.set(xlabel = 'number of datapoints', ylabel = 'run time')
ax.errorbar(times0[:, 0], times0[:, 1], times0[:, 2], label = 'PCA', linewidth=2)
ax.errorbar(times1[:, 0], times1[:, 1], times1[:, 2], label = 'PCA_high_dim', linewidth=2)
ax.legend();

%time PCA(Xbar, 2)
%time PCA_high_dim(Xbar, 2)
pass
