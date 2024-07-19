# Dependencies
import timeit

import numpy as np
import pandas as pd

from load_data import load_mnist

from ipywidgets import interact

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline


# Load MNIST dataset
MNIST = load_mnist()
images, labels = MNIST['data'], MNIST['target']


# Quick plot of an image from the dataset to make sure data loaded properly
plt.figure(figsize = (4,4))
plt.imshow(images[0].reshape(28,28), cmap = 'gray');


'''
The next four functions preprocess the data so the images can have a zero mean and 
variance of one. The pixel encodings are originally uint8 and must be converted to 
a floating point number between 0-1. The mean mu, μ, must be subtracted from each image.
Each image must be scaled dimensionally by 1/σ, where σ is the standard deviation,
X_normalized will be the normalized dataset.
'''

def normalize(X):
    '''
    Normalize the dataset X.

    Parameters:
    X: ndarray of shape (N, D)

    Returns:
    X_normalized: ndarray of shape (N, D) - normalized dataset
    mean: ndarray of shape (D,) - mean of each feature
    std_dev: ndarray of shape (D,) - standard deviation of each feature
    
     In the future, you could encounter dimensions where the standard deviation is
     zero, for those when you do normalization the normalized data will be NaN. 
     Handle this by setting using `std = 1` for those dimensions when doing normalization.
    '''
    # Calculate the mean of each feature
    mu = np.mean(X, axis = 0) 
    # Calculate standard deviation of each feature
    std_dev = np.std(X, axis = 0)
    # Create copy of std_dev
    std_dev_filled = std_dev.copy()
    # Avoid division by zero by setting std_dev to 1 where it is 0
    std_dev_filled[std == 0] = 1.0
    # Normalize the dataset
    X_normalized = (X - mu) / std_dev_filled                  
    return X_normalized, mu, std_dev


def compute_eigen(S):
    ''' 
    Parameters:
    S: ndarray of shape (D, D)

    Returns:
    eigen_values: ndarray of shape (D,) - sorted eigenavlues
    eigen_vectors: ndarray of shape (D, D) - corresponding eigenvectors
     
    Computes the eigenvalues and corresponding eigenvectors for the covariance matrix S 
    and sorts them in descending order of eigenvalues.
    '''
    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(S)
    # Get indices of sorted eigenvalues in descending order
    sorted_indices = eigen_values.argsort()[:: - 1]
    # Sort eigenvalues
    eigen_values = eigen_values[sorted_indices]
    # Sort eigenvectors
    eigen_vectors = eigen_vectors[:, sorted_indices]
    return (eigen_values, eigen_vectors) 


def projection_matrix(B):
   '''
   Parameters:
   B = ndarray of shape (D, M) is the basis for the subspace 
   
   Returns:
   P = projection matrix of shape (D, D)
   
   Computes the projection matrix onto the space spanned by B.
   '''
   # Compute projection matrix
    return B @ (np.linalg.inv(B.T @ B)) @ B.T
 

def pca(X, num_components):
   '''
   Parameters:
   X: ndarray of size (N, D), where N is the number of datapoints and D is the 
   dimension of the data.
   
   num_components: int - number of principal components to use

   Returns:
   X_reconstructed: ndarray of shape (N, D) the reconstruction of X from the 
   first `num_components` principal components

   Perform principal component analysis (PCA) on dataset X.
   '''
    # Normalize the dataset
    X_normalized, mu, std_dev = normalize(X)
    # Compute covariance matrix
    eigen_values, eigen_vectors = np.cov(X_normalized, rowvar = False, bias = True)
    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = compute_eigens(covariance_matrix)
    # Compute projection matrix
    projection_matrix = compute_projection_matrix(eigen_vectors[:, :num_components])
    # Reconstructed dataset
    X_reconstructed = (projection_matrix @ X_normalized.T).T
    return X_reconstruct 


 def mse(predicted, actual):
    '''
    Parameters:
       predicted: ndarray of shape (N, D)
       actual: ndarray of shape (N, D)

    Returns: mse: float - mean squared error
    
    Computes the mean squared error (MSE) between predicted and actual data. In other
    words: how many principal components are needed in order to reach a mean squared 
    error of less than 100 for the dataset.
    '''
    # Calculate and return MSE
    return np.square(predicted - actual).sum(axis = 1).mean()


def pca_high_dimensional(X, num_components):
   '''
   Parameters:
   X: ndarray of size (N, D), where D is the dimension of the sample, N is 
   the number of samples.
   
   num_components: int - number of principal components to use
   
   Returns:
   X: ndarray of shape (N, D) the reconstruction of X from the first `num_components` 
   pricipal components.

   Performs PCA for high-dimensional datasets where the number of features is larger 
   than the number of samples (small sample size). When the dimensionality of the dataset 
   is larger than the number of given samples, PCA can be implemented in a more optimized 
   way for high-dimensionality.
   '''
   # Get dimensions of the dataset
   N, D = X.shape
   # Compute Gram matrix
   gram_matrix = (X @ X.T) / N
   # Compute eigenvalues and eigenvectors
   eigen_values, eigen_vectors = compute_eigen(gram_matrix)
   # Compute principle components
   principal_components = (X.T @ eigen_vectors)[:, :num_components]
   # Compute projection matrix
   projection_matrix = compute_projection_matrix(principal_components)
   # Reconstruct dataset
   X_reconstructed = (projection_matrix @ X.T).T
   return X_reconstruct 


def time(func, repeat=10):
   '''
   Parameters:
      func: function - function to measure time
      repeat: int - number of repetitions

   Return:
   mean_time: float - mean execution time
   std_time: float - standard deviation of execution time

   Measure the execution time of a function. Finds time/space complexity by 
   comparing the running time between pca and pca_high_dimensional functions.
   '''
   # Initialize list to store execution times
   times = []
   # Repeat measurement
   for _ in range(repeat):
      # Start timer
      start = timeit.default_timer()
      # Execute function
      func()
      # Stop timer
      stop = timeit.default_timer()
      # Append elaspsed time to list
      times.append(stop - start)
   # Return mean and standard deviation
   return np.mean(times), np.std(times)


# Data preprocessing
# Define number of datapoints to use
NUM_DATAPOINTS = 1000
# Reshape and normalize images
X = (images.reshape(-1, 28 * 28)[:NUM_DATAPOINTS]) / 255.0
print(X.shape)
# Normalize X
X_normalized, mu, std_dev = normalize(X)
print(mu.shape)
print(std_de.shape)


# The number of principal components we use, the smaller the reconstruction error will be
# Iterate PCA from sklearn
for num_component in range(1, 20):
    # Compare custom PCA implementation with sklean's implementation
    from sklearn.decomposition import PCA as SKPCA
    # Initialize sklearn PCA
    pca = SKPCA(n_components = num_component, svd_solver = 'full')
    # Reconstruct using sklearn PCA
    sklearn_reconstructed = pca.inverse_transform(pca.fit_transform(X_normalized))
    # Reconstruct using custom PCA
    custom_reconstructed = PCA(X_normalized, num_component)
    # Assert almost equal
    np.testing.assert_almost_equal(custom_reconstructed, sklearn_reconstructed)
    print(np.square(reconst - sklearn_reconst).sum())

# Evaluate reconstruction error for different number of principal components
# Initialize list to store errors
reconstruction_errors = []
# Initialize list to store reconstructions
reconstructions = []

# LEFT OFF HERE
# LEFT OFF HERE
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
